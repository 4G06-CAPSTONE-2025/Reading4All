import { useState } from "react";
import "./login.css";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // UI states
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // ✅ Placeholder: replace this with your real API call later
  async function loginApiPlaceholder({ email, password }) {
    // Simulate a network delay
    await new Promise((r) => setTimeout(r, 500));

    // For now: always fail so you can see the failure UI
    return { ok: false, error: "Login failed. Please try again." };
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg("");

    // Basic front-end validation (feel free to tweak)
    if (!email.trim() || !password) {
      setErrorMsg("Please enter your email and password.");
      return;
    }

    setIsLoading(true);
    try {
      const res = await loginApiPlaceholder({ email, password });
      if (!res.ok) {
        setErrorMsg(res.error || "Login failed. Please try again.");
        return;
      }

      // Later you’ll navigate to /upload after real auth succeeds
      navigate("/upload");
    } catch (err) {
      setErrorMsg("Something went wrong. Please retry.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-page">
      {/* Top heading section (logo is already in your Header) */}
      <div className="auth-heading">
        <h1>Alternative Text Generation</h1>
        <p>Generate clear, concise alternative text for STEM diagrams</p>
      </div>

      {/* Card */}
      <div className="auth-card">
        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="auth-label" htmlFor="email">
            Email
          </label>
          <input
            id="email"
            className="auth-input"
            type="email"
            placeholder="student@mcmaster.ca"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
          />

          <label className="auth-label" htmlFor="password">
            Password
          </label>
          <input
            id="password"
            className="auth-input"
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
          />

          <button className="auth-button" type="submit" disabled={isLoading}>
            {isLoading ? "Logging in..." : "Login"}
          </button>
        </form>

        {/* Failure state UI (matches your screenshot style) */}
        {errorMsg && (
          <div className="auth-error">
            <div className="error-icon" aria-hidden="true">
              !
            </div>
            <div className="error-text">{errorMsg}</div>
          </div>
        )}
      </div>
    </div>
  );
}
