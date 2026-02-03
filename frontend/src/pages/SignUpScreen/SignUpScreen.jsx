import { useMemo, useState } from "react";
import "./SignUpScreen.css";
import { useNavigate } from "react-router-dom";

export default function SignUpScreen() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");

  const [otp, setOtp] = useState("");

  const [isSending, setIsSending] = useState(false);
  const [isSigningUp, setIsSigningUp] = useState(false);

  const [verificationSent, setVerificationSent] = useState(false);
  const [isVerified, setIsVerified] = useState(false);

  const [errorMsg, setErrorMsg] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

  const emailOk = useMemo(() => email.trim().toLowerCase().endsWith("@mcmaster.ca"), [email]);

  // ✅ MOCK: "Send verification code" (pretend we emailed OTP = 123456)
  async function sendVerificationMock() {
    await new Promise((r) => setTimeout(r, 500));
    return { ok: true };
  }

  // ✅ MOCK: verify OTP
  async function verifyOtpMock(code) {
    await new Promise((r) => setTimeout(r, 400));
    return { ok: code === "123456" };
  }

  // ✅ MOCK: signup
  async function signUpMock() {
    await new Promise((r) => setTimeout(r, 500));
    return { ok: true };
  }

  const handleSendVerification = async () => {
    setErrorMsg("");
    setSuccessMsg("");
    setIsVerified(false);

    if (!email.trim()) {
      setErrorMsg("Email is required.");
      return;
    }
    if (!emailOk) {
      setErrorMsg("Please use your McMaster email (@mcmaster.ca).");
      return;
    }
    if (!pw || pw.length < 6) {
      setErrorMsg("Password must be at least 6 characters.");
      return;
    }
    if (pw !== confirmPw) {
      setErrorMsg("Passwords do not match.");
      return;
    }

    setIsSending(true);
    try {
      const res = await sendVerificationMock();
      if (!res.ok) {
        setErrorMsg("Could not send verification code. Retry.");
        return;
      }
      setVerificationSent(true);
      setSuccessMsg("Verification code sent. (Mock OTP: 123456)");
    } catch {
      setErrorMsg("Could not send verification code. Retry.");
    } finally {
      setIsSending(false);
    }
  };

  const handleSignUp = async (e) => {
    e.preventDefault();
    setErrorMsg("");
    setSuccessMsg("");

    if (!verificationSent) {
      setErrorMsg("Please send a verification code first.");
      return;
    }

    setIsSigningUp(true);
    try {
      const verify = await verifyOtpMock(otp.trim());
      if (!verify.ok) {
        setIsVerified(false);
        setErrorMsg("Verification Code Incorrect! Retry");
        return;
      }

      setIsVerified(true);

      const signup = await signUpMock();
      if (!signup.ok) {
        setErrorMsg("Sign up failed. Please retry.");
        return;
      }

      setSuccessMsg("Verification Code Correct! Sign Up Successful");
    } catch {
      setErrorMsg("Sign up failed. Please retry.");
    } finally {
      setIsSigningUp(false);
    }
  };

  const signUpDisabled = !verificationSent || isSigningUp;

  return (
    <div className="auth-page">
      <div className="auth-heading">
        <h1>Alternative Text Generation</h1>
        <p>Generate clear, concise alternative text for STEM diagrams</p>
      </div>

      <div className="auth-card signup-card">
        <form className="auth-form" onSubmit={handleSignUp}>
          <h2 className="signup-title">Sign up using McMaster Email</h2>

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
            value={pw}
            onChange={(e) => setPw(e.target.value)}
          />

          <label className="auth-label">Confirm Password</label>
          <input
            className="auth-input"
            type="password"
            placeholder="••••••••"
            value={confirmPw}
            onChange={(e) => setConfirmPw(e.target.value)}
          />

          <button
            type="button"
            className="auth-button"
            onClick={handleSendVerification}
            disabled={isSending}
          >
            {isSending ? "Sending..." : "Send Verification Code"}
          </button>

          <div className="otp-block">
            <div className="otp-label">Enter One Time Verification Code</div>
            <input
              className="auth-input otp-input"
              type="text"
              inputMode="numeric"
              placeholder="123456"
              value={otp}
              onChange={(e) => setOtp(e.target.value)}
              disabled={!verificationSent}
            />
          </div>

          <button className="auth-button" type="submit" disabled={signUpDisabled}>
            {isSigningUp ? "Signing up..." : "Sign Up"}
          </button>

          {/* Success message + Login button (like your screenshot) */}
          {isVerified && successMsg && (
            <>
              <div className="signup-success">{successMsg}</div>
              <button
                type="button"
                className="auth-button"
                onClick={() => navigate("/login")}
              >
                Login
              </button>
            </>
          )}
        </form>

        {/* Error bubble (only for error state like your failure screenshot) */}
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
