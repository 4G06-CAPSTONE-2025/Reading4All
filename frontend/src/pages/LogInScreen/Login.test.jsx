import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import Login from "./login";
import { MemoryRouter } from "react-router-dom";

// mock navigate
const mockNavigate = jest.fn();

jest.mock("react-router-dom", () => ({
  ...jest.requireActual("react-router-dom"),
  useNavigate: () => mockNavigate,
}));

beforeEach(() => {
  global.fetch = jest.fn();
  mockNavigate.mockClear();
});

afterEach(() => {
  jest.clearAllMocks();
});

describe("Login component", () => {

  test("renders login form", () => {
    render(
      <MemoryRouter>
        <Login />
      </MemoryRouter>
    );

    expect(screen.getByText("Physics Alternative Text Generation")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("johndoe@email.com")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("••••••••")).toBeInTheDocument();
  });

  test("shows validation error if fields are empty", async () => {
    render(
      <MemoryRouter>
        <Login />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByText("Login"));

    expect(
      await screen.findByText("Email and password are required.")
    ).toBeInTheDocument();
  });

  test("allows typing email and password", () => {
    render(
      <MemoryRouter>
        <Login />
      </MemoryRouter>
    );

    const emailInput = screen.getByPlaceholderText("johndoe@email.com");
    const passwordInput = screen.getByPlaceholderText("••••••••");

    fireEvent.change(emailInput, {
      target: { value: "test@email.com" },
    });

    fireEvent.change(passwordInput, {
      target: { value: "password123" },
    });

    expect(emailInput.value).toBe("test@email.com");
    expect(passwordInput.value).toBe("password123");
  });

  test("successful login navigates to upload page", async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });

    render(
      <MemoryRouter>
        <Login />
      </MemoryRouter>
    );

    fireEvent.change(screen.getByPlaceholderText("johndoe@email.com"), {
      target: { value: "test@email.com" },
    });

    fireEvent.change(screen.getByPlaceholderText("••••••••"), {
      target: { value: "password123" },
    });

    fireEvent.click(screen.getByText("Login"));

    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith("/upload")
    );
  });

  test("shows backend error message", async () => {
    fetch.mockResolvedValueOnce({
      ok: false,
      json: async () => ({
        error: "Invalid credentials",
      }),
    });

    render(
      <MemoryRouter>
        <Login />
      </MemoryRouter>
    );

    fireEvent.change(screen.getByPlaceholderText("johndoe@email.com"), {
      target: { value: "test@email.com" },
    });

    fireEvent.change(screen.getByPlaceholderText("••••••••"), {
      target: { value: "password123" },
    });

    fireEvent.click(screen.getByText("Login"));

    expect(
      await screen.findByText("Invalid credentials")
    ).toBeInTheDocument();
  });

  test("shows network error message", async () => {
    fetch.mockRejectedValueOnce(new Error("network error"));

    render(
      <MemoryRouter>
        <Login />
      </MemoryRouter>
    );

    fireEvent.change(screen.getByPlaceholderText("johndoe@email.com"), {
      target: { value: "test@email.com" },
    });

    fireEvent.change(screen.getByPlaceholderText("••••••••"), {
      target: { value: "password123" },
    });

    fireEvent.click(screen.getByText("Login"));

    expect(
      await screen.findByText("Something went wrong. Please try again.")
    ).toBeInTheDocument();
  });

});