import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import RequireAuth from "./RequireAuth";

global.fetch = jest.fn();

function renderComponent(child = <div>Protected Content</div>) {
  render(
    <MemoryRouter>
      <RequireAuth>{child}</RequireAuth>
    </MemoryRouter>
  );
}

describe("RequireAuth", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("renders nothing while loading", () => {
    fetch.mockResolvedValueOnce({ ok: true });

    const { container } = render(
      <MemoryRouter>
        <RequireAuth>
          <div>Protected Content</div>
        </RequireAuth>
      </MemoryRouter>
    );

    expect(screen.queryByText("Protected Content")).not.toBeInTheDocument();
  });

  test("renders children when authenticated", async () => {
    fetch.mockResolvedValueOnce({ ok: true });

    renderComponent();

    expect(
      await screen.findByText("Protected Content")
    ).toBeInTheDocument();
  });

  test("redirects to login when not authenticated", async () => {
    fetch.mockResolvedValueOnce({ ok: false });

    render(
      <MemoryRouter initialEntries={["/protected"]}>
        <RequireAuth>
          <div>Protected Content</div>
        </RequireAuth>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.queryByText("Protected Content")).not.toBeInTheDocument();
    });
  });
});