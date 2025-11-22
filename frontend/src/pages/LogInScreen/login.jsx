import React, { useState } from "react";
import "./login.css";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");

    // temporary login for POC
    const fakeEmail = "student@mcmaster.ca";
    const fakePassword = "password";

    if (email === fakeEmail && password === fakePassword) {
      window.location.href = "/main";
    } else {
      setError("Invalid email or password.");
    }
  };

  return (
    <div className="login-container">
      <h2 className="login-title">Sign In</h2>

      <form className="login-form" onSubmit={handleSubmit}>
        <label htmlFor="email" className="login-label">
          Email
        </label>
        <input
          id="email"
          type="email"
          aria-label="Email address"
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
          aria-label="Password"
          className="login-input"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        {error && <p className="login-error">{error}</p>}

        <button type="submit" className="login-button">
          Log In
        </button>
      </form>
    </div>
  );
}
