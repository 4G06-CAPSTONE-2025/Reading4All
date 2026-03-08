import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import HistoryPage from "./HistoryPage";

beforeEach(() => {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      ok: true,
      json: async () => ({}),
    })
  );
});

afterEach(() => {
  jest.clearAllMocks();
});

test("renders the history page title", () => {
  fetch.mockResolvedValueOnce({
    ok: true,
    json: async () => ({
      history: [],
    }),
  });

  render(<HistoryPage />);

  expect(screen.getByText(/Alt Text History/i)).toBeInTheDocument();
});

test("loads and displays history items", async () => {
  const mockHistory = [
    {
      image: "base64image",
      altText: "A cat sitting on a couch",
    },
  ];

  fetch.mockResolvedValueOnce({
    ok: true,
    json: async () => ({
      history: mockHistory,
    }),
  });

  render(<HistoryPage />);

  expect(
    await screen.findByText("A cat sitting on a couch")
  ).toBeInTheDocument();
});

test("shows message when no history exists", async () => {
  fetch.mockResolvedValueOnce({
    ok: true,
    json: async () => ({
      history: [],
    }),
  });

  render(<HistoryPage />);

  await waitFor(() =>
    expect(
      screen.getByText("No alternative text has been generated yet!")
    ).toBeInTheDocument()
  );
});

test("shows error if history fails to load", async () => {
  fetch.mockRejectedValueOnce(new Error("API error"));

  render(<HistoryPage />);

  await waitFor(() =>
    expect(
      screen.getByText("Failed to Load History. Please try again!")
    ).toBeInTheDocument()
  );
});

test("copies all alt text to clipboard", async () => {
  const mockHistory = [
    { image: "img1", altText: "First alt text" },
    { image: "img2", altText: "Second alt text" },
  ];

  fetch.mockResolvedValueOnce({
    ok: true,
    json: async () => ({
      history: mockHistory,
    }),
  });

  Object.assign(navigator, {
    clipboard: {
      writeText: jest.fn(),
    },
  });

  render(<HistoryPage />);

  const copyAllButton = await screen.findByText("Copy All Alt Texts");

  fireEvent.click(copyAllButton);

  expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
    "First alt text\n\nSecond alt text"
  );
});

test("copies individual alt text", async () => {
  const mockHistory = [{ image: "img1", altText: "A dog running" }];

  fetch.mockResolvedValueOnce({
    ok: true,
    json: async () => ({
      history: mockHistory,
    }),
  });

  Object.assign(navigator, {
    clipboard: {
      writeText: jest.fn(),
    },
  });

  render(<HistoryPage />);

  const copyButton = await screen.findByText("Copy Alt Text");

  fireEvent.click(copyButton);

  expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
    "A dog running"
  );
});

test("shows error if clipboard copy fails", async () => {
  const mockHistory = [{ image: "img1", altText: "Some alt text" }];

  fetch.mockResolvedValueOnce({
    ok: true,
    json: async () => ({
      history: mockHistory,
    }),
  });

  Object.assign(navigator, {
    clipboard: {
      writeText: jest.fn().mockRejectedValue(new Error("copy failed")),
    },
  });

  render(<HistoryPage />);

  const copyButton = await screen.findByText("Copy Alt Text");

  fireEvent.click(copyButton);

  await waitFor(() =>
    expect(
      screen.getByText("Failed to copy to clipboard.")
    ).toBeInTheDocument()
  );
});
