package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

const (
	baseURL          = "https://api.openai.com/v1"
	pollInterval     = 5 * time.Second
	defaultMaxTokens = 1000
	defaultModel     = "gpt-4o"
	slackWebhookURL  = "https://hooks.slack.com/services/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"
)

// BatchRequest represents a single request in the batch
type BatchRequest struct {
	CustomID string      `json:"custom_id"`
	Method   string      `json:"method"`
	URL      string      `json:"url"`
	Body     RequestBody `json:"body"`
}

type RequestBody struct {
	Model     string                         `json:"model"`
	Messages  []openai.ChatCompletionMessage `json:"messages"`
	MaxTokens int                            `json:"max_tokens"`
}

// BatchResponse represents the response from batch creation
type BatchResponse struct {
	ID           string `json:"id"`
	Status       string `json:"status"`
	OutputFileID string `json:"output_file_id,omitempty"`
}

func main() {
	apiKey := os.Getenv("OPENAI_API_TRANSLATE_KEY")
	if apiKey == "" {
		fmt.Fprintf(os.Stderr, "Error: OPENAI_API_TRANSLATE_KEY environment variable is not set\n")
		os.Exit(1)
	}

	client := openai.NewClient(apiKey)
	ctx := context.Background()
	httpClient := &http.Client{}

	if err := processBatch(ctx, client, httpClient); err != nil {
		fmt.Fprintf(os.Stderr, "Error processing batch: %v\n", err)
		os.Exit(1)
	}
}

func processBatch(ctx context.Context, client *openai.Client, httpClient *http.Client) error {
	fmt.Println("=== Starting batch processing ===")
	fmt.Println("1. Creating batch requests...")

	requests := []BatchRequest{
		{
			CustomID: "request-3",
			Method:   "POST",
			URL:      "/v1/chat/completions",
			Body: RequestBody{
				Model: defaultModel,
				Messages: []openai.ChatCompletionMessage{
					{Role: "system", Content: "You are a helpful assistant."},
					{Role: "user", Content: "Hello world!"},
				},
				MaxTokens: defaultMaxTokens,
			},
		},
		{
			CustomID: "request-4",
			Method:   "POST",
			URL:      "/v1/chat/completions",
			Body: RequestBody{
				Model: defaultModel,
				Messages: []openai.ChatCompletionMessage{
					{Role: "system", Content: "You are a helpful assistant."},
					{Role: "user", Content: "what is stock?"},
				},
				MaxTokens: defaultMaxTokens,
			},
		},
	}

	fmt.Println("2. Creating JSONL input file...")
	if err := createJSONLFile("batch_input.jsonl", requests); err != nil {
		return fmt.Errorf("creating JSONL file: %w", err)
	}
	fmt.Println("✓ JSONL file created successfully")

	fmt.Println("\n3. Uploading file to OpenAI...")
	uploadedFile, err := client.CreateFile(ctx, openai.FileRequest{
		FileName: "batch_input.jsonl",
		FilePath: "batch_input.jsonl",
		Purpose:  "batch",
	})
	if err != nil {
		return fmt.Errorf("uploading file: %w", err)
	}
	fmt.Printf("✓ File uploaded successfully with ID: %s\n", uploadedFile.ID)

	fmt.Println("\n4. Creating batch process...")
	batch, err := createBatch(ctx, httpClient, uploadedFile.ID)
	if err != nil {
		return fmt.Errorf("creating batch: %w", err)
	}
	fmt.Printf("✓ Batch created successfully with ID: %s\n", batch.ID)

	fmt.Println("\n5. Monitoring batch status...")
	return monitorBatchStatus(ctx, httpClient, batch.ID)
}

func createJSONLFile(filename string, data interface{}) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	if arr, ok := data.([]BatchRequest); ok {
		for _, item := range arr {
			if err := encoder.Encode(item); err != nil {
				return err
			}
		}
	}
	return nil
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

	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/batches", bytes.NewBuffer(reqBody))
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

func sendSlackNotification(message string) error {
	payload := struct {
		Text string `json:"text"`
	}{
		Text: message,
	}

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshaling slack payload: %w", err)
	}

	resp, err := http.Post(slackWebhookURL, "application/json", bytes.NewBuffer(jsonPayload))
	if err != nil {
		return fmt.Errorf("sending slack notification: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("slack notification failed with status: %d", resp.StatusCode)
	}

	return nil
}

func monitorBatchStatus(ctx context.Context, httpClient *http.Client, batchID string) error {
	fmt.Println("Starting batch status monitoring...")
	attempts := 1
	startTime := time.Now()

	for {
		status, err := getBatchStatus(ctx, httpClient, batchID)
		if err != nil {
			return fmt.Errorf("checking batch status: %w", err)
		}

		fmt.Printf("[Attempt %d] Batch status: %s\n", attempts, status.Status)

		switch status.Status {
		case "completed":
			elapsedTime := time.Since(startTime)
			fmt.Println("\n✓ Batch processing completed!")
			fmt.Println("Downloading results...")
			err := downloadResults(ctx, httpClient, status.OutputFileID)
			if err != nil {
				return fmt.Errorf("downloading results: %w", err)
			}
			fmt.Println("✓ Results downloaded successfully to batch_output.jsonl")

			message := fmt.Sprintf("✅ 배치 처리가 완료되었습니다!\n배치 ID: %s\n총 소요 시간: %s",
				batchID,
				formatDuration(elapsedTime),
			)
			if err := sendSlackNotification(message); err != nil {
				fmt.Printf("Warning: Failed to send Slack notification: %v\n", err)
			}

			return nil
		case "failed":
			return fmt.Errorf("batch processing failed")
		case "expired":
			return fmt.Errorf("batch processing expired")
		default:
			fmt.Printf("Waiting %v before next status check...\n", pollInterval)
			time.Sleep(pollInterval)
			attempts++
		}
	}
}

func getBatchStatus(ctx context.Context, httpClient *http.Client, batchID string) (*BatchResponse, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("%s/batches/%s", baseURL, batchID), nil)
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

func downloadResults(ctx context.Context, httpClient *http.Client, fileID string) error {
	fmt.Printf("Downloading file with ID: %s\n", fileID)
	req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("%s/files/%s/content", baseURL, fileID), nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENAI_API_TRANSLATE_KEY"))

	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	content, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	fmt.Printf("Writing results to batch_output.jsonl...\n")
	if err := os.WriteFile("batch_output.jsonl", content, 0644); err != nil {
		return fmt.Errorf("writing output file: %w", err)
	}
	return nil
}

func formatDuration(d time.Duration) string {
	d = d.Round(time.Second)
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second

	if h > 0 {
		return fmt.Sprintf("%d시간 %d분 %d초", h, m, s)
	} else if m > 0 {
		return fmt.Sprintf("%d분 %d초", m, s)
	}
	return fmt.Sprintf("%d초", s)
}
