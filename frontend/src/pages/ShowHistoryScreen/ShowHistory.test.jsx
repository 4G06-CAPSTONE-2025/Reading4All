import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import ShowHistory from  "./ShowHistory";

const mockNavigate = jest.fn();

const mockHistory = [
  {
    id: 1,
    name: "diagram.png",
    date: "2025-03-10",
    altText: "A physics diagram showing forces",
    fileUrl: "https://example.com/image.png",
  },
];

jest.mock("react-router-dom", () => ({
  ...jest.requireActual("react-router-dom"),
  useNavigate: () => mockNavigate,
  useLocation: () => ({
    state: { history: mockHistory },
  }),
}));

describe("ShowHistory", () => {

  beforeEach(() => {
    mockNavigate.mockClear();
  });

  test("renders page title and back button", () => {
    render(<ShowHistory />);

    expect(screen.getByText("Full Session History")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Return to the main screen/i })).toBeInTheDocument();
  });

  test("displays history entries", () => {
    render(<ShowHistory />);

    expect(screen.getByText("diagram.png")).toBeInTheDocument();
    expect(screen.getByText("A physics diagram showing forces")).toBeInTheDocument();
  });

  test("clicking history entry shows preview", () => {
    render(<ShowHistory />);

    const entry = screen.getByText("diagram.png");

    fireEvent.click(entry);

    expect(screen.getByRole("img")).toBeInTheDocument();
    expect(screen.getByRole("img")).toBeInTheDocument();
expect(screen.getByAltText("A physics diagram showing forces")).toBeInTheDocument();
  });

  test("back button navigates to upload page", () => {
    render(<ShowHistory />);

    const backButton = screen.getByText("Back to Home");

    fireEvent.click(backButton);

    expect(mockNavigate).toHaveBeenCalledWith("/upload");
  });

});
