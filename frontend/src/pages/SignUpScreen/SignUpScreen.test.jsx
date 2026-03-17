import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import SignUpScreen from "./SignUpScreen";

global.fetch = jest.fn();

const renderComponent = () => {
  render(
    <MemoryRouter>
      <SignUpScreen />
    </MemoryRouter>
  );
};

describe("SignUpScreen", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("renders signup form", () => {
    renderComponent();

    expect(screen.getByText(/create an account/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/enter your email/i)).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText(/minimum 8 characters/i)
    ).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText(/re-enter password/i)
    ).toBeInTheDocument();
  });

  test("shows error if email is missing", async () => {
    renderComponent();

    const button = screen.getByRole("button", {
      name: /send verification code/i,
    });

    await userEvent.click(button);

    expect(await screen.findByText(/email is required/i)).toBeInTheDocument();
  });

  test("shows error if password is too short", async () => {
    renderComponent();

    await userEvent.type(
      screen.getByPlaceholderText(/enter your email/i),
      "test@test.com"
    );

    await userEvent.type(
      screen.getByPlaceholderText(/minimum 8 characters/i),
      "123"
    );

    await userEvent.click(
      screen.getByRole("button", { name: /send verification code/i })
    );

    expect(
      await screen.findByText(/password must be at least 8 characters/i)
    ).toBeInTheDocument();
  });

  test("shows error if passwords do not match", async () => {
    renderComponent();

    await userEvent.type(
      screen.getByPlaceholderText(/enter your email/i),
      "test@test.com"
    );

    await userEvent.type(
      screen.getByPlaceholderText(/minimum 8 characters/i),
      "12345678"
    );

    await userEvent.type(
      screen.getByPlaceholderText(/re-enter password/i),
      "87654321"
    );

    await userEvent.click(
      screen.getByRole("button", { name: /send verification code/i })
    );

    expect(
      await screen.findByText(/passwords do not match/i)
    ).toBeInTheDocument();
  });

  test("sends verification code successfully", async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });

    renderComponent();

    await userEvent.type(
      screen.getByPlaceholderText(/enter your email/i),
      "test@test.com"
    );

    await userEvent.type(
      screen.getByPlaceholderText(/minimum 8 characters/i),
      "12345678"
    );

    await userEvent.type(
      screen.getByPlaceholderText(/re-enter password/i),
      "12345678"
    );

    await userEvent.click(
      screen.getByRole("button", { name: /send verification code/i })
    );

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));

    const otpInput = screen.getByPlaceholderText("123456");

    expect(otpInput).not.toBeDisabled();
  });

  test("shows error when OTP is missing during signup", async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });

    renderComponent();

    await userEvent.type(
      screen.getByPlaceholderText(/enter your email/i),
      "test@test.com"
    );

    await userEvent.type(
      screen.getByPlaceholderText(/minimum 8 characters/i),
      "12345678"
    );

    await userEvent.type(
      screen.getByPlaceholderText(/re-enter password/i),
      "12345678"
    );

    await userEvent.click(
      screen.getByRole("button", { name: /send verification code/i })
    );

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));

    await userEvent.click(screen.getByRole("button", { name: /sign up/i }));

    expect(
      await screen.findByText(/verification code is required/i)
    ).toBeInTheDocument();
  });

  test("completes successful signup", async () => {
    fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

    renderComponent();

    await userEvent.type(
      screen.getByPlaceholderText(/enter your email/i),
      "test@test.com"
    );

    await userEvent.type(
      screen.getByPlaceholderText(/minimum 8 characters/i),
      "12345678"
    );

    await userEvent.type(
      screen.getByPlaceholderText(/re-enter password/i),
      "12345678"
    );

    await userEvent.click(
      screen.getByRole("button", { name: /send verification code/i })
    );

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));

    await userEvent.type(screen.getByPlaceholderText("123456"), "123456");

    await userEvent.click(screen.getByRole("button", { name: /sign up/i }));

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));

    expect(
      await screen.findByText(/verification successful/i)
    ).toBeInTheDocument();
  });
});