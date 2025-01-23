package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"sort"

	"bufio"

	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

const (
	INPUT_FILE = "./csv/transltion_32b.csv"

	MAX_RETRIES = 10
	RETRY_DELAY = 5 * time.Second
	MAX_WORKERS = 10

	// Logging control flags
	LOG_CSV_READ_DETAILS    = true  // CSV 파일 읽기 과정 상세 로그
	LOG_EVALUATION_DETAILS  = false // 번역 평가 과정 상세 로그
	LOG_CSV_WRITE_DETAILS   = false // CSV 출력 과정 상세 로그
	LOG_TRANSLATION_CONTENT = false // 번역 내용 상세 로그 (긴 텍스트)

	SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/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" // Add this to .env file instead
)

var (
	TARGET_LANGUAGE_LIST []string // Will be populated from CSV headers
	logger               *zap.SugaredLogger
	OUTPUT_FILE          string // Will be set in main()
	timestamp            string // Add this line to store the timestamp globally
)

var combinedPrompt = `
# You are given an original Korean text and its translation. Please evaluate both the translation quality and classify the text category.

## Original Text (Korean):
%s

## Translation :
%s

### Task 1 - Error Type Classification:
Please evaluate the translation and provide the results in the following format:

Error Type: Must be one of:
- "정상": Translation effectively conveys the original meaning. Minor stylistic differences or slight word choice variations are acceptable as long as they don't impact the core message
- "의미전달미흡": Severe mistranslation that fundamentally changes the original meaning
- "단어변경": Significant word choice errors that notably alter the intended meaning
- "추가/누락": Critical information is added or omitted, changing the message substantially

Note: Focus on whether the translation maintains the essential meaning. Minor variations in style or word choice that preserve the core message should be classified as "정상".

### Task 2 - Category Classification:
Please classify the original Korean text into one of the following categories:
Category:
1. "은행 & 금융": 은행에서 사용되는 금융과 관련이 깊은 문장
2. "비은행 & 금융": 은행에서 사용되지 않으나 일상에서 사용되는 금융과 관련이 깊은 문장
3. "비은행 & 비금융": 은행에서 사용되지 않고 금융과 관련이 아예 없는 문장

# Provide the results in JSON format only:
{
    "category": "category_here",
    "error_type": "error_type_here"
}
`

type TranslationData struct {
	QuestionIndex string
	Question      string
	Translations  map[string]string // Simplified to just store translations
}

type ErrorRow struct {
	LineNumber int
	Content    []string
	Error      string
}

type CombinedEvaluation struct {
	OriginalKr  string `json:"original_kr"`
	Language    string `json:"language"`
	Translation string `json:"translation"`
	Category    string `json:"category"`
	ErrorType   string `json:"error_type"`
}

var languageNameMap = map[string]string{
	"en":    "영어",
	"zh-cn": "중국어_간체",
	"zh-tw": "중국어_번체",
	"ja":    "일본어",
	"vi":    "베트남어",
	"uz":    "우즈베크어",
	"ru":    "러시아어",
	"ne":    "네팔어",
	"bn":    "벵골어",
	"th":    "태국어",
	"id":    "인도네시아어",
	"tl":    "타갈로그어",
	"si":    "싱할라어",
	"my":    "미얀마어",
	"km":    "크메르어",
	"mn":    "몽골어",
}

type BatchRequest struct {
	CustomID string      `json:"custom_id"`
	Method   string      `json:"method"`
	URL      string      `json:"url"`
	Body     RequestBody `json:"body"`
}

type RequestBody struct {
	Model    string                         `json:"model"`
	Messages []openai.ChatCompletionMessage `json:"messages"`
}

type BatchResponse struct {
	ID           string `json:"id"`
	Status       string `json:"status"`
	OutputFileID string `json:"output_file_id,omitempty"`
}

// Add new structs for JSON logging
type CSVReadLog struct {
	Timestamp      string            `json:"timestamp"`
	InputFile      string            `json:"input_file"`
	TotalRecords   int               `json:"total_records"`
	LanguageCounts map[string]int    `json:"language_counts"`
	Data           []TranslationData `json:"data"`
}

type EvaluationLog struct {
	Timestamp     string                        `json:"timestamp"`
	QuestionIndex string                        `json:"question_index"`
	Results       map[string]CombinedEvaluation `json:"results"`
}

type CSVWriteLog struct {
	Timestamp  string                                   `json:"timestamp"`
	OutputFile string                                   `json:"output_file"`
	Results    map[string]map[string]CombinedEvaluation `json:"results"`
	Data       []TranslationData                        `json:"data"`
}

func init() {
	// Create logs directory if it doesn't exist
	err := os.MkdirAll("logs", os.ModePerm)
	if err != nil {
		log.Fatalf("Error creating logs directory: %v", err)
	}

	// Create a custom encoder configuration
	encoderConfig := zap.NewProductionEncoderConfig()
	encoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder

	// Create a custom zap configuration
	config := zap.Config{
		Level:            zap.NewAtomicLevelAt(zap.InfoLevel),
		Development:      false,
		Encoding:         "json",
		EncoderConfig:    encoderConfig,
		OutputPaths:      []string{"stdout", "logs/evaluation.log"},
		ErrorOutputPaths: []string{"stderr"},
	}

	// Build the logger
	zapLogger, err := config.Build()
	if err != nil {
		log.Fatalf("Error building zap logger: %v", err)
	}

	// Create a SugaredLogger
	logger = zapLogger.Sugar()
}

func main() {
	// Calculate timestamp once at the start
	timestamp = time.Now().Format("20060102_150405")

	// Create logs directory if it doesn't exist
	err := os.MkdirAll("logs", os.ModePerm)
	if err != nil {
		logger.Fatalf("Error creating logs directory: %v", err)
	}

	// Use timestamp for output file
	OUTPUT_FILE = fmt.Sprintf("./csv/classification_32b_%s.csv", timestamp)

	// Sync the logger when main() exits
	defer logger.Sync()

	// Load .env file
	err = godotenv.Load()
	if err != nil {
		logger.Fatalf("Error loading .env file: %v", err)
	}

	// Verify API key is set
	apiKey := os.Getenv("OPENAI_API_TRANSLATE_KEY")
	if apiKey == "" {
		logger.Fatal("OPENAI_API_TRANSLATE_KEY is not set in .env file")
	}

	translationData, err := readCSV(INPUT_FILE)
	if err != nil {
		logger.Fatalf("Error reading input file: %v", err)
	}

	// Sort translationData by QuestionIndex
	sort.Slice(translationData, func(i, j int) bool {
		return translationData[i].QuestionIndex < translationData[j].QuestionIndex
	})

	// Prepare batch requests
	var requests []BatchRequest
	for _, data := range translationData {
		for lang, translation := range data.Translations {
			languageName := languageNameMap[lang]
			if languageName == "" {
				languageName = lang
			}
			prompt := fmt.Sprintf(combinedPrompt, data.Question, translation)

			request := BatchRequest{
				CustomID: fmt.Sprintf("%s_%s", data.QuestionIndex, lang),
				Method:   "POST",
				URL:      "/v1/chat/completions",
				Body: RequestBody{
					Model: "gpt-4o",
					Messages: []openai.ChatCompletionMessage{
						{Role: "user", Content: prompt},
					},
				},
			}
			requests = append(requests, request)
		}
	}

	// Use timestamp for input file - Updated path
	inputFileName := fmt.Sprintf("logs/batch_input_%s.jsonl", timestamp)
	if err := createJSONLFile(inputFileName, requests); err != nil {
		logger.Fatalf("Error creating JSONL file: %v", err)
	}

	// Upload file to OpenAI
	client := openai.NewClient(os.Getenv("OPENAI_API_TRANSLATE_KEY"))
	uploadedFile, err := client.CreateFile(context.Background(), openai.FileRequest{
		FileName: inputFileName,
		FilePath: inputFileName,
		Purpose:  "batch",
	})
	if err != nil {
		logger.Fatalf("Error uploading file: %v", err)
	}

	// Create batch
	httpClient := &http.Client{}
	batch, err := createBatch(context.Background(), httpClient, uploadedFile.ID)
	if err != nil {
		logger.Fatalf("Error creating batch: %v", err)
	}

	// Initialize results map
	results := make(map[string]map[string]CombinedEvaluation)

	// Create a temporary map for the batch results
	batchResults := make(map[string]CombinedEvaluation)

	// Send initial Slack notification
	message := fmt.Sprintf("Translation evaluation batch started!\nBatch ID: %s", batch.ID)
	if err := sendSlackNotification(message); err != nil {
		logger.Errorf("Error sending Slack notification: %v", err)
	}

	logger.Infof("Monitoring batch status for ID: %s", batch.ID)

	// Monitor batch status and wait for completion
	if err := monitorBatchStatus(context.Background(), httpClient, batch.ID, batchResults); err != nil {
		logger.Fatalf("Error monitoring batch: %v", err)
	}

	// Process batch results into the final format
	for customID, eval := range batchResults {
		parts := strings.Split(customID, "_")
		if len(parts) != 2 {
			continue
		}
		questionIndex, lang := parts[0], parts[1]

		if results[questionIndex] == nil {
			results[questionIndex] = make(map[string]CombinedEvaluation)
		}
		results[questionIndex][lang] = eval
	}

	// Write results to CSV
	err = writeCSV(OUTPUT_FILE, results, translationData)
	if err != nil {
		logger.Fatalf("Error writing output file: %v", err)
	}

	// Send completion notification
	completionMessage := fmt.Sprintf("Translation evaluation complete!\nResults written to: %s", OUTPUT_FILE)
	if err := sendSlackNotification(completionMessage); err != nil {
		logger.Errorf("Error sending completion notification: %v", err)
	}

	logger.Infof("Process completed successfully. Results written to: %s", OUTPUT_FILE)
}

func readCSV(filename string) ([]TranslationData, error) {
	if LOG_CSV_READ_DETAILS {
		logger.Infof("Starting to read CSV file: %s", filename)
	}
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("error opening file %s: %v", filename, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = '\t'
	reader.LazyQuotes = true

	headers, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("error reading headers: %v", err)
	}
	if LOG_CSV_READ_DETAILS {
		logger.Infof("CSV Headers: %v", headers)
	}

	// Extract language information from headers
	TARGET_LANGUAGE_LIST = []string{}
	languageColumns := make(map[string]int)

	for i := 1; i < len(headers); i++ {
		header := headers[i]
		if strings.HasSuffix(header, "_번역문") {
			lang := strings.TrimSuffix(header, "_번역문")
			TARGET_LANGUAGE_LIST = append(TARGET_LANGUAGE_LIST, lang)
			languageColumns[lang] = i
			if LOG_CSV_READ_DETAILS {
				logger.Infof("Detected language column: %s at index %d", lang, i)
			}
		}
	}

	var translationData []TranslationData
	lineNum := 1
	languageCounts := make(map[string]int)

	logger.Info("Starting to read CSV records...")
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			logger.Errorf("Error reading line %d: %v", lineNum, err)
			continue
		}

		data := TranslationData{
			QuestionIndex: strconv.Itoa(lineNum),
			Question:      record[0],
			Translations:  make(map[string]string),
		}

		for lang, colIndex := range languageColumns {
			if colIndex < len(record) && record[colIndex] != "" {
				data.Translations[lang] = record[colIndex]
				languageCounts[lang]++
				if LOG_TRANSLATION_CONTENT {
					logger.Debugf("Line %d - %s translation: %s", lineNum, lang, record[colIndex])
				}
			}
		}

		translationData = append(translationData, data)
		lineNum++
	}

	logger.Infof("Finished reading CSV. Total records: %d", len(translationData))
	logger.Info("Translation counts by language:")

	// Create log structure
	readLog := CSVReadLog{
		Timestamp:      time.Now().Format(time.RFC3339),
		InputFile:      filename,
		TotalRecords:   len(translationData),
		LanguageCounts: languageCounts,
		Data:           translationData,
	}

	// Write JSON log
	if err := writeJSONLog(readLog, "csv_read"); err != nil {
		logger.Errorf("Error writing CSV read log: %v", err)
	}

	return translationData, nil
}

func writeErrorList(errorRows []ErrorRow) error {
	file, err := os.Create("error_list.csv")
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write headers
	headers := []string{"Line Number", "Content", "Error"}
	if err := writer.Write(headers); err != nil {
		return err
	}

	// Write error rows
	for _, row := range errorRows {
		record := []string{
			strconv.Itoa(row.LineNumber),
			strings.Join(row.Content, "|"),
			row.Error,
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil
}

func evaluateTranslations(data TranslationData) (map[string]CombinedEvaluation, error) {
	if LOG_EVALUATION_DETAILS {
		logger.Infof("Starting evaluation for QuestionIndex: %s", data.QuestionIndex)
	}

	// 배치 요청 준비
	var requests []BatchRequest
	for lang, translation := range data.Translations {
		languageName := languageNameMap[lang]
		if languageName == "" {
			languageName = lang
		}
		prompt := fmt.Sprintf(combinedPrompt, languageName, data.Question, languageName, translation)

		request := BatchRequest{
			CustomID: fmt.Sprintf("%s_%s", data.QuestionIndex, lang),
			Method:   "POST",
			URL:      "/v1/chat/completions",
			Body: RequestBody{
				Model: "gpt-4",
				Messages: []openai.ChatCompletionMessage{
					{Role: "user", Content: prompt},
				},
			},
		}
		requests = append(requests, request)
	}

	// JSONL 파일 생성
	inputFileName := fmt.Sprintf("batch_input_%s.jsonl", time.Now().Format("20060102_150405"))
	if err := createJSONLFile(inputFileName, requests); err != nil {
		return nil, fmt.Errorf("creating JSONL file: %w", err)
	}

	// 파일 업로드
	client := openai.NewClient(os.Getenv("OPENAI_API_TRANSLATE_KEY"))
	uploadedFile, err := client.CreateFile(context.Background(), openai.FileRequest{
		FileName: inputFileName,
		FilePath: inputFileName,
		Purpose:  "batch",
	})
	if err != nil {
		return nil, fmt.Errorf("uploading file: %w", err)
	}

	// 배치 생성
	httpClient := &http.Client{}
	batch, err := createBatch(context.Background(), httpClient, uploadedFile.ID)
	if err != nil {
		return nil, fmt.Errorf("creating batch: %w", err)
	}

	// 배치 상태 모니터링
	results := make(map[string]CombinedEvaluation)
	if err := monitorBatchStatus(context.Background(), httpClient, batch.ID, results); err != nil {
		return nil, fmt.Errorf("monitoring batch: %w", err)
	}

	// Create log structure
	evalLog := EvaluationLog{
		Timestamp:     time.Now().Format(time.RFC3339),
		QuestionIndex: data.QuestionIndex,
		Results:       results,
	}

	// Write JSON log
	if err := writeJSONLog(evalLog, "evaluation"); err != nil {
		logger.Errorf("Error writing evaluation log: %v", err)
	}

	return results, nil
}

func createBatch(ctx context.Context, httpClient *http.Client, fileID string) (*BatchResponse, error) {
	reqBody, err := json.Marshal(struct {
		InputFileID      string `json:"input_file_id"`
		Endpoint         string `json:"endpoint"`
		CompletionWindow string `json:"completion_window"`
	}{
		InputFileID:      fileID,
		Endpoint:         "/v1/chat/completions",
		CompletionWindow: "24h",
	})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/batches", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENAI_API_TRANSLATE_KEY"))
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var batch BatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&batch); err != nil {
		return nil, err
	}
	return &batch, nil
}

// Helper function to get translation or return empty string if not supported
func getTranslation(translations map[string]string, lib string) string {
	return translations[lib]
}

func parseTranslationEvaluation(content string) (CombinedEvaluation, error) {
	var evaluation CombinedEvaluation

	// Extract JSON from content
	regex := regexp.MustCompile(`(?s)\{.*\}`)
	match := regex.FindString(content)
	if match == "" {
		return evaluation, fmt.Errorf("no JSON found in content")
	}

	// Parse JSON
	err := json.Unmarshal([]byte(match), &evaluation)
	if err != nil {
		return evaluation, fmt.Errorf("error parsing JSON: %v", err)
	}

	// Validate error type
	validErrorTypes := map[string]bool{
		"정상":     true,
		"의미전달미흡": true,
		"단어변경":   true,
		"추가/누락":  true,
	}
	if !validErrorTypes[evaluation.ErrorType] {
		return evaluation, fmt.Errorf("invalid error type: %s", evaluation.ErrorType)
	}

	// Validate category
	validCategories := map[string]bool{
		"은행 & 금융":   true,
		"비은행 & 금융":  true,
		"비은행 & 비금융": true,
	}
	if !validCategories[evaluation.Category] {
		return evaluation, fmt.Errorf("invalid category: %s", evaluation.Category)
	}

	return evaluation, nil
}

// Modify writeCSV function
func writeCSV(filename string, results map[string]map[string]CombinedEvaluation, data []TranslationData) error {
	if LOG_CSV_WRITE_DETAILS {
		logger.Infof("Starting to write results to CSV: %s", filename)
	}
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("error creating file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	writer.Comma = '\t'
	defer writer.Flush()

	logger.Info("Writing headers...")
	headers := []string{"원문_한국어", "언어", "문장유형", "번역문_언어", "평가유형"}
	if err := writer.Write(headers); err != nil {
		return fmt.Errorf("error writing headers: %v", err)
	}

	logger.Info("Processing data rows...")
	for _, item := range data {
		logger.Infof("Processing QuestionIndex: %s", item.QuestionIndex)

		for _, lang := range TARGET_LANGUAGE_LIST {
			translation := item.Translations[lang]
			var errorType, category string

			if evalResult, exists := results[item.QuestionIndex]; exists {
				if eval, hasLang := evalResult[lang]; hasLang {
					errorType = eval.ErrorType
					category = eval.Category // Use Category from evaluation results
				}
			}

			languageName := languageNameMap[lang]
			if languageName == "" {
				languageName = lang
			}

			record := []string{
				item.Question,
				languageName,
				category, // Use category for 문장유형
				translation,
				errorType, // Use errorType for 평가유형
			}

			if err := writer.Write(record); err != nil {
				return fmt.Errorf("error writing record: %v", err)
			}
		}
	}

	logger.Info("Successfully completed writing results to CSV")

	// Create log structure
	writeLog := CSVWriteLog{
		Timestamp:  time.Now().Format(time.RFC3339),
		OutputFile: filename,
		Results:    results,
		Data:       data,
	}

	// Write JSON log
	if err := writeJSONLog(writeLog, "csv_write"); err != nil {
		logger.Errorf("Error writing CSV write log: %v", err)
	}

	return nil
}

// Add this helper function to format duration
func formatDuration(d time.Duration) string {
	d = d.Round(time.Second)
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second
	return fmt.Sprintf("%02d:%02d:%02d", h, m, s)
}

func callGPTAPI(prompt string) (string, error) {
	client := openai.NewClient(os.Getenv("OPENAI_API_TRANSLATE_KEY"))
	resp, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model: openai.GPT4,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
	})
	if err != nil {
		return "", err
	}
	return resp.Choices[0].Message.Content, nil
}

func getBatchStatus(ctx context.Context, httpClient *http.Client, batchID string) (*BatchResponse, error) {
	req, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("https://api.openai.com/v1/batches/%s", batchID), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENAI_API_TRANSLATE_KEY"))

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var status BatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return nil, err
	}
	return &status, nil
}

func downloadResults(ctx context.Context, httpClient *http.Client, fileID string, outputFileName string) error {
	req, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("https://api.openai.com/v1/files/%s/content", fileID), nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENAI_API_TRANSLATE_KEY"))

	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	file, err := os.Create(outputFileName)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	return err
}

func createJSONLFile(filename string, requests []BatchRequest) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	for _, req := range requests {
		data, err := json.Marshal(req)
		if err != nil {
			return err
		}
		if _, err := writer.Write(data); err != nil {
			return err
		}
		if _, err := writer.WriteString("\n"); err != nil {
			return err
		}
	}
	return writer.Flush()
}

// Add new function for Slack notifications
func sendSlackNotification(message string) error {
	payload := map[string]string{
		"text": message,
	}

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("error marshaling payload: %v", err)
	}

	resp, err := http.Post(os.Getenv("SLACK_WEBHOOK_URL"), "application/json", bytes.NewBuffer(jsonPayload))
	if err != nil {
		return fmt.Errorf("error sending notification: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return nil
}

func writeJSONLog(data interface{}, prefix string) error {
	filename := fmt.Sprintf("logs/%s_%s.json", prefix, timestamp)
	jsonData, err := json.MarshalIndent(data, "", "    ")
	if err != nil {
		return fmt.Errorf("error marshaling JSON: %v", err)
	}

	return os.WriteFile(filename, jsonData, 0644)
}

func monitorBatchStatus(ctx context.Context, httpClient *http.Client, batchID string, results map[string]CombinedEvaluation) error {
	retries := 0
	for {
		status, err := getBatchStatus(ctx, httpClient, batchID)
		if err != nil {
			if retries < MAX_RETRIES {
				retries++
				time.Sleep(RETRY_DELAY)
				continue
			}
			return fmt.Errorf("checking batch status: %w", err)
		}

		switch status.Status {
		case "completed":
			// Updated output file path
			outputFileName := fmt.Sprintf("logs/batch_output_%s.jsonl", timestamp)
			if err := downloadResults(ctx, httpClient, status.OutputFileID, outputFileName); err != nil {
				return fmt.Errorf("downloading results: %w", err)
			}

			if err := parseResults(outputFileName, results); err != nil {
				return fmt.Errorf("parsing results: %w", err)
			}

			return nil
		case "failed":
			return fmt.Errorf("batch processing failed")
		case "expired":
			return fmt.Errorf("batch processing expired")
		default:
			fmt.Printf("Current status: %s. Waiting %v before next check...\n",
				status.Status, RETRY_DELAY)
			time.Sleep(RETRY_DELAY)
		}
	}
}

func parseResults(filename string, results map[string]CombinedEvaluation) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var batchResponse struct {
			CustomID string `json:"custom_id"`
			Response struct {
				Body struct {
					Choices []struct {
						Message struct {
							Content string `json:"content"`
						} `json:"message"`
					} `json:"choices"`
				} `json:"body"`
			} `json:"response"`
		}

		if err := json.Unmarshal(scanner.Bytes(), &batchResponse); err != nil {
			return fmt.Errorf("error unmarshaling batch response: %w", err)
		}

		if len(batchResponse.Response.Body.Choices) > 0 {
			content := batchResponse.Response.Body.Choices[0].Message.Content
			// Remove markdown code block markers if present
			content = strings.TrimPrefix(content, "```json\n")
			content = strings.TrimSuffix(content, "\n```")

			var evaluation CombinedEvaluation
			if err := json.Unmarshal([]byte(content), &evaluation); err != nil {
				return fmt.Errorf("error unmarshaling evaluation content: %w", err)
			}

			// Validate error type
			validErrorTypes := map[string]bool{
				"정상":     true,
				"의미전달미흡": true,
				"단어변경":   true,
				"추가/누락":  true,
			}
			if !validErrorTypes[evaluation.ErrorType] {
				return fmt.Errorf("invalid error type: %s", evaluation.ErrorType)
			}

			// Validate category
			validCategories := map[string]bool{
				"은행 & 금융":   true,
				"비은행 & 금융":  true,
				"비은행 & 비금융": true,
			}
			if !validCategories[evaluation.Category] {
				return fmt.Errorf("invalid category: %s", evaluation.Category)
			}

			results[batchResponse.CustomID] = evaluation
		}
	}

	return scanner.Err()
}
