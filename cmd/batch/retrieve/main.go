package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"go.uber.org/zap"
)

const (
	MAX_RETRIES = 10000
	RETRY_DELAY = 5 * time.Second
)

// 필요한 구조체 정의 추가
type CombinedEvaluation struct {
	OriginalKr  string `json:"original_kr"` // 원문_한국어
	Language    string `json:"language"`    // 언어
	Category    string `json:"category"`    // 문장유형
	Translation string `json:"translation"` // 번역문_언어
	ErrorType   string `json:"error_type"`  // 평가유형
}

type BatchResponse struct {
	Status       string `json:"status"`
	OutputFileID string `json:"output_file_id"`
}

func main() {
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		fmt.Printf("Error loading .env file: %v\n", err)
		return
	}

	// Check if batch ID is provided
	if len(os.Args) != 2 {
		fmt.Println("Usage: go run retrieve_results.go <batch_id>")
		return
	}
	batchID := os.Args[1]

	// Initialize logger
	logger, err := initLogger()
	if err != nil {
		fmt.Printf("Error initializing logger: %v\n", err)
		return
	}
	defer logger.Sync()

	// Create HTTP client
	httpClient := &http.Client{}

	// Monitor batch status and get results
	timestamp := time.Now().Format("20060102_150405")
	results := make(map[string]CombinedEvaluation)

	logger.Info("Starting to monitor batch status...")
	if err := monitorBatchStatus(context.Background(), httpClient, batchID, results); err != nil {
		logger.Fatalf("Error monitoring batch: %v", err)
		return
	}

	// Process results and write to CSV
	outputFile := fmt.Sprintf("./csv/classification_results_%s.csv", timestamp)
	if err := writeResultsToCSV(outputFile, results); err != nil {
		logger.Fatalf("Error writing results to CSV: %v", err)
		return
	}

	logger.Infof("Results successfully written to: %s", outputFile)

	// Send Slack notification
	slackWebhookURL := os.Getenv("SLACK_WEBHOOK_URL")
	if slackWebhookURL == "" {
		logger.Warn("SLACK_WEBHOOK_URL is not set, skipping Slack notification")
	} else {
		message := fmt.Sprintf("Batch results processing complete!\nResults written to: %s", outputFile)
		if err := sendSlackNotification(slackWebhookURL, message); err != nil {
			logger.Errorf("Error sending Slack notification: %v", err)
		}
	}
}

// Update sendSlackNotification function to accept webhookURL parameter
func sendSlackNotification(webhookURL, message string) error {
	if webhookURL == "" {
		return fmt.Errorf("slack webhook URL is empty")
	}

	payload := map[string]string{
		"text": message,
	}

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("error marshaling payload: %v", err)
	}

	resp, err := http.Post(webhookURL, "application/json", bytes.NewBuffer(jsonPayload))
	if err != nil {
		return fmt.Errorf("error sending notification: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return nil
}

func initLogger() (*zap.SugaredLogger, error) {
	config := zap.NewProductionConfig()
	config.OutputPaths = []string{"stdout", "logs/retrieve_results.log"}

	zapLogger, err := config.Build()
	if err != nil {
		return nil, err
	}

	return zapLogger.Sugar(), nil
}

func writeResultsToCSV(filename string, results map[string]CombinedEvaluation) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("error creating file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	writer.Comma = '\t'
	defer writer.Flush()

	// Write headers
	headers := []string{"원문_한국어", "언어", "문장유형", "번역문_언어", "평가유형"}
	if err := writer.Write(headers); err != nil {
		return fmt.Errorf("error writing headers: %v", err)
	}

	// Write results
	for _, eval := range results {
		record := []string{
			strings.ReplaceAll(eval.OriginalKr, "\t", " "),
			strings.ReplaceAll(eval.Language, "\t", " "),
			strings.ReplaceAll(eval.Category, "\t", " "),
			strings.ReplaceAll(eval.Translation, "\t", " "),
			strings.ReplaceAll(eval.ErrorType, "\t", " "),
		}
		if err := writer.Write(record); err != nil {
			return fmt.Errorf("error writing record: %v", err)
		}
	}

	return nil
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
			results[batchResponse.CustomID] = evaluation
		}
	}

	return scanner.Err()
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
			timestamp := time.Now().Format("20060102_150405")
			outputFileName := fmt.Sprintf("batch_output_%s.jsonl", timestamp)
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

func parseTranslationEvaluationFromString(content string) (CombinedEvaluation, error) {
	// 문자열에서 필요한 정보를 추출하는 로직
	lines := strings.Split(content, "\n")
	evaluation := CombinedEvaluation{}

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "error_type:") {
			evaluation.ErrorType = strings.TrimSpace(strings.TrimPrefix(line, "error_type:"))
		} else if strings.HasPrefix(line, "original_kr:") {
			evaluation.OriginalKr = strings.TrimSpace(strings.TrimPrefix(line, "original_kr:"))
		} else if strings.HasPrefix(line, "language:") {
			evaluation.Language = strings.TrimSpace(strings.TrimPrefix(line, "language:"))
		} else if strings.HasPrefix(line, "category:") {
			evaluation.Category = strings.TrimSpace(strings.TrimPrefix(line, "category:"))
		} else if strings.HasPrefix(line, "translation:") {
			evaluation.Translation = strings.TrimSpace(strings.TrimPrefix(line, "translation:"))
		}
	}

	return evaluation, nil
}
