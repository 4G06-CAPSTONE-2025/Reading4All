import "./homeScreen.css";
import { useNavigate } from "react-router-dom";

export default function HomeScreen() {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      <h1 className="home-title">Alternative Text Generation</h1>
      <p className="home-subtitle">
        Generate clear, concise alternative text for STEM diagrams
      </p>

      <div className="home-buttons">
        <button
          className="primary-btn"
          onClick={() => navigate("/signup")}
        >
          First Time Here? Sign up using McMaster Email
        </button>

        <button
          className="secondary-btn"
          onClick={() => navigate("/login")}
        >
          Login using McMaster Email
        </button>
      </div>
    </div>
  );
}
