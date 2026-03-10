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

  /* ---------- LOOK & FEEL / STYLE TESTS ---------- */

  test("page title is visible and clearly displayed", () => {
    render(<UploadScreen />);

    expect(
      screen.getByText(/physics alternative text generation/i)
    ).toBeInTheDocument();
  });

  test("page subtitle is visible to guide users", () => {
    render(<UploadScreen />);

    expect(
      screen.getByText(/generate clear, concise alternative text/i)
    ).toBeInTheDocument();
  });
  test("upload instructions are accessible through aria label", () => {
    render(<UploadScreen />);
  
    expect(
      screen.getByLabelText(/press enter or space to browse/i)
    ).toBeInTheDocument();
  });

  test("upload area has accessible label", () => {
    render(<UploadScreen />);

    expect(
      screen.getByLabelText(/upload image for alt text generation/i)
    ).toBeInTheDocument();
  });

  /* ---------- FR1 : IMAGE UPLOAD ---------- */

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

  /* ---------- FR3 : ALT TEXT GENERATION ---------- */

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

    await userEvent.click(await screen.findByText(/generate alt text/i));

    expect(
      await screen.findByDisplayValue(/physics diagram of a pendulum/i)
    ).toBeInTheDocument();
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

  /* ---------- USER INTERACTION ---------- */

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

  test("keyboard enter opens file input", () => {
    render(<UploadScreen />);

    const dropBox = screen.getByLabelText(/press enter or space to browse/i);
    const input = screen.getByLabelText(/upload image for alt text generation/i);

    const clickSpy = jest.spyOn(input, "click");

    fireEvent.keyDown(dropBox, { key: "Enter" });

    expect(clickSpy).toHaveBeenCalled();
  });

  /* ---------- ERROR HANDLING ---------- */

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

});