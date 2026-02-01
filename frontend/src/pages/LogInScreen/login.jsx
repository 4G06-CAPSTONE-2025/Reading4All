import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./login.css";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");

    const fakeEmail = "student@mcmaster.ca";
    const fakePassword = "password";

    if (email === fakeEmail && password === fakePassword) {
      navigate("/upload");
    } else {
      setError("Invalid email or password.");
    }
  };

  return (
    <div className="login-container" aria-label="Log in form">
      <h2 className="login-title">Sign In</h2>

      <form className="login-form" onSubmit={handleSubmit}>
        <label htmlFor="email" className="login-label">
          Email
        </label>
        <input
          id="email"
          type="email"
          aria-label="Enter email address"
          className="login-input"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />

        <label htmlFor="password" className="login-label">
          Password
        </label>
        <input
          id="password"
          type="password"
          aria-label="Enter password"
          className="login-input"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        {error && <p className="login-error">{error}</p>}

        <button
          type="submit"
          className="login-button"
          aria-label="Button for logging in to a session"
        >
          Log In to Session
        </button>
      </form>
    </div>
  );
}
