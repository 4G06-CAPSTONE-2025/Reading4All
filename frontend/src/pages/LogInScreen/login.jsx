import { useState } from "react";
import "./login.css";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // 🔹 MOCK LOGIN API
  async function loginApiMock({ email, password }) {
    await new Promise((r) => setTimeout(r, 500));

    // Domain check
    if (!email.endsWith("@mcmaster.ca")) {
      return {
        ok: false,
        error: "Please use your McMaster email (@mcmaster.ca).",
      };
    }

    // Hardcoded credentials (for now)
    if (email === "student@mcmaster.ca" && password === "student") {
      return { ok: true };
    }

    return {
      ok: false,
      error: "Invalid email or password. Retry.",
    };
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg("");

    if (!email.trim() || !password) {
      setErrorMsg("Email and password are required.");
      return;
    }

    setIsLoading(true);
    try {
      const res = await loginApiMock({ email, password });

      if (!res.ok) {
        setErrorMsg(res.error);
        return;
      }

      // ✅ Successful login
      navigate("/upload");
    } catch {
      setErrorMsg("Something went wrong. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-heading">
        <h1>Alternative Text Generation</h1>
        <p>Generate clear, concise alternative text for STEM diagrams</p>
      </div>

      <div className="auth-card">
        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="auth-label">Email</label>
          <input
            className="auth-input"
            type="email"
            placeholder="student@mcmaster.ca"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          <label className="auth-label">Password</label>
          <input
            className="auth-input"
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          <button className="auth-button" type="submit" disabled={isLoading}>
            {isLoading ? "Logging in..." : "Login"}
          </button>
        </form>

        {errorMsg && (
          <div className="auth-error">
            <div className="error-icon">!</div>
            <div className="error-text">{errorMsg}</div>
          </div>
        )}
      </div>
    </div>
  );
}
