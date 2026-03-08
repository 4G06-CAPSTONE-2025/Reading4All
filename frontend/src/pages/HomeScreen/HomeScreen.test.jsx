import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { MemoryRouter } from "react-router-dom";
import HomeScreen from "./homeScreen";

describe("HomeScreen", () => {

  test("renders homepage title and subtitle", () => {
    render(
      <MemoryRouter>
        <HomeScreen />
      </MemoryRouter>
    );

    expect(
      screen.getByText("Physics Alternative Text Generation")
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        "Generate clear, concise alternative text for physics diagrams"
      )
    ).toBeInTheDocument();
  });

  test("signup button exists", () => {
    render(
      <MemoryRouter>
        <HomeScreen />
      </MemoryRouter>
    );

    const signupButton = screen.getByText(
      "First Time Here? Sign up using McMaster Email"
    );

    expect(signupButton).toBeInTheDocument();
  });

  test("login button exists", () => {
    render(
      <MemoryRouter>
        <HomeScreen />
      </MemoryRouter>
    );

    const loginButton = screen.getByText(
      "Login using McMaster Email"
    );

    expect(loginButton).toBeInTheDocument();
  });

});