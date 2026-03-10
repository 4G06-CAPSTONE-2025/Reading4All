import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import UploadScreen from "./UploadScreen";

global.fetch = jest.fn();

Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn(),
  },
});

global.URL.createObjectURL = jest.fn(() => "preview-url");
global.URL.revokeObjectURL = jest.fn();

describe("UploadScreen", () => {

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("renders page title and subtitle", () => {
    render(<UploadScreen />);

    expect(
      screen.getByText(/physics alternative text generation/i)
    ).toBeInTheDocument();

    expect(
      screen.getByText(/generate clear, concise alternative text/i)
    ).toBeInTheDocument();
  });

  test("choose file button is visible", () => {
    render(<UploadScreen />);

    expect(screen.getByText(/choose file/i)).toBeInTheDocument();
  });

  test("uploads image successfully", async () => {
    fetch.mockResolvedValueOnce({ ok: true });

    render(<UploadScreen />);

    const file = new File(["image"], "diagram.png", { type: "image/png" });

    const input = screen.getByLabelText(/upload image for alt text generation/i);

    await userEvent.upload(input, file);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalled();
    });

    expect(
      screen.getByText(/successfully uploaded image/i)
    ).toBeInTheDocument();
  });

  test("generates alt text after upload", async () => {
    fetch
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          alt_text: "A physics diagram of a pendulum.",
          entry_id: 1,
        }),
      });

    render(<UploadScreen />);

    const file = new File(["image"], "diagram.png", { type: "image/png" });

    const input = screen.getByLabelText(/upload image for alt text generation/i);

    await userEvent.upload(input, file);

    expect(
      await screen.findByText(/successfully uploaded image/i)
    ).toBeInTheDocument();

    await userEvent.click(screen.getByText(/generate alt text/i));

    expect(
      await screen.findByDisplayValue(/physics diagram of a pendulum/i)
    ).toBeInTheDocument();
  });

  test("copy alt text button copies text", async () => {
    fetch
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          alt_text: "Example alt text",
          entry_id: 10,
        }),
      });

    render(<UploadScreen />);

    const file = new File(["image"], "diagram.png", { type: "image/png" });

    const input = screen.getByLabelText(/upload image for alt text generation/i);

    await userEvent.upload(input, file);

    await userEvent.click(await screen.findByText(/generate alt text/i));

    const copyButton = await screen.findByText(/copy alt text/i);

    await userEvent.click(copyButton);

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      "Example alt text"
    );
  });

  test("removes uploaded image", async () => {
    fetch.mockResolvedValueOnce({ ok: true });

    render(<UploadScreen />);

    const file = new File(["image"], "diagram.png", { type: "image/png" });

    const input = screen.getByLabelText(/upload image for alt text generation/i);

    await userEvent.upload(input, file);

    const removeButton = await screen.findByLabelText(/remove uploaded image/i);

    await userEvent.click(removeButton);

    expect(
      screen.queryByText(/successfully uploaded image/i)
    ).not.toBeInTheDocument();
  });

  test("drag and drop uploads image", async () => {
    fetch.mockResolvedValueOnce({ ok: true });

    render(<UploadScreen />);

    const file = new File(["image"], "diagram.png", { type: "image/png" });

    const dropBox = screen.getByLabelText(/press enter or space to browse/i);

    fireEvent.drop(dropBox, {
      dataTransfer: { files: [file] },
    });

    await waitFor(() => {
      expect(fetch).toHaveBeenCalled();
    });
  });
  
  test("dragging over upload box adds dragging class", () => {
    render(<UploadScreen />);
  
    const dropBox = screen.getByLabelText(/press enter or space to browse/i);
  
    fireEvent.dragOver(dropBox);
  
    expect(dropBox.className).toMatch(/upload-frame-dragging/i);
  });
  
  test("drag leave removes dragging class", () => {
    render(<UploadScreen />);
  
    const dropBox = screen.getByLabelText(/press enter or space to browse/i);
  
    fireEvent.dragOver(dropBox);
    fireEvent.dragLeave(dropBox);
  
    expect(dropBox.className).not.toMatch(/upload-frame-dragging/i);
  });
  
  test("keyboard enter opens file input", () => {
    render(<UploadScreen />);
  
    const dropBox = screen.getByLabelText(/press enter or space to browse/i);
    const input = screen.getByLabelText(/upload image for alt text generation/i);
  
    const clickSpy = jest.spyOn(input, "click");
  
    fireEvent.keyDown(dropBox, { key: "Enter" });
  
    expect(clickSpy).toHaveBeenCalled();
  });
  
  test("shows upload error from backend", async () => {
    fetch.mockResolvedValueOnce({
      ok: false,
      json: async () => ({ error: "FILE_SIZE_INVALID" }),
    });
  
    render(<UploadScreen />);
  
    const file = new File(["image"], "large.png", { type: "image/png" });
    const input = screen.getByLabelText(/upload image for alt text generation/i);
  
    await userEvent.upload(input, file);
  
    expect(
      await screen.findByText(/image size exceeds 10 megabytes/i)
    ).toBeInTheDocument();
  });
  
  test("alt text generation failure shows error", async () => {
    fetch
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce({ ok: false });
  
    render(<UploadScreen />);
  
    const file = new File(["image"], "diagram.png", { type: "image/png" });
    const input = screen.getByLabelText(/upload image for alt text generation/i);
  
    await userEvent.upload(input, file);
  
    await userEvent.click(await screen.findByText(/generate alt text/i));
  
    expect(
      await screen.findByText(/failed to generate alt text/i)
    ).toBeInTheDocument();
  });
  
  test("copy alt text failure shows error", async () => {
    navigator.clipboard.writeText.mockRejectedValueOnce();
  
    fetch
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          alt_text: "Example alt text",
          entry_id: 5,
        }),
      });
  
    render(<UploadScreen />);
  
    const file = new File(["image"], "diagram.png", { type: "image/png" });
    const input = screen.getByLabelText(/upload image for alt text generation/i);
  
    await userEvent.upload(input, file);
  
    await userEvent.click(await screen.findByText(/generate alt text/i));
  
    const copyButton = await screen.findByText(/copy alt text/i);
  
    await userEvent.click(copyButton);
  
    expect(
      await screen.findByText(/failed to copy to clipboard/i)
    ).toBeInTheDocument();
  });
  
  test("editing alt text and saving sends API request", async () => {
    fetch
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          alt_text: "Original text",
          entry_id: 99,
        }),
      })
      .mockResolvedValueOnce({ ok: true });
  
    render(<UploadScreen />);
  
    const file = new File(["image"], "diagram.png", { type: "image/png" });
    const input = screen.getByLabelText(/upload image for alt text generation/i);
  
    await userEvent.upload(input, file);
  
    await userEvent.click(await screen.findByText(/generate alt text/i));
  
    const textarea = await screen.findByDisplayValue(/original text/i);
  
    await userEvent.clear(textarea);
    await userEvent.type(textarea, "Edited alt text");
  
    const saveButton = screen.getByText(/save edits/i);
  
    await userEvent.click(saveButton);
  
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining("edit-alt-text"),
        expect.any(Object)
      );
    });
  });
  
  test("uploading new image resets generated alt text", async () => {
    fetch.mockResolvedValue({ ok: true });
  
    render(<UploadScreen />);
  
    const file1 = new File(["image"], "diagram1.png", { type: "image/png" });
    const file2 = new File(["image"], "diagram2.png", { type: "image/png" });
  
    const input = screen.getByLabelText(/upload image for alt text generation/i);
  
    await userEvent.upload(input, file1);
    await userEvent.upload(input, file2);
  
    expect(screen.getByText("diagram2.png")).toBeInTheDocument();
  });
  
});