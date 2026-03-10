import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { MemoryRouter } from "react-router-dom";
import HomeScreen from "./homeScreen";

describe("HomeScreen", () => {

  /* ---------- LOOK & FEEL ---------- */

  test("homepage title is clearly displayed", () => {
    render(
      <MemoryRouter>
        <HomeScreen />
      </MemoryRouter>
    );

    expect(
      screen.getByText("Physics Alternative Text Generation")
    ).toBeInTheDocument();
  });

  test("homepage subtitle explains system purpose", () => {
    render(
      <MemoryRouter>
        <HomeScreen />
      </MemoryRouter>
    );

    expect(
      screen.getByText(
        "Generate clear, concise alternative text for physics diagrams"
      )
    ).toBeInTheDocument();
  });

  /* ---------- STYLE / USABILITY ---------- */

  test("signup button exists and is visible", () => {
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

  test("login button exists and is visible", () => {
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