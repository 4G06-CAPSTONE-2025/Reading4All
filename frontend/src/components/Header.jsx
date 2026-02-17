import "./Header.css";
import mac_logo from "./mcmaster-logo.png";
import { useLocation, useNavigate } from "react-router-dom";

export default function Header() {
  const location = useLocation();
  const navigate = useNavigate();

  const currLocation = location.pathname;

  const handleSignOut = async () => {
    try {
      await fetch(
        "https://reading4all-backend.onrender.com/api/logout/",
        {
          method: "POST",
          credentials: "include",
        }
      );
    } finally {
      navigate("/");
    }
  };

  return (
    <header className="Reading4All-header">
      <div className="header-navigation">
        {currLocation === "/upload" && (
          <button
            className="navigation-button"
            onClick={() => navigate("/session-history")}
          >
            View History
          </button>
        )}


            {
                (currLocation==="/session-history" || currLocation==="/signup" ||  currLocation==="/login") && (
                    <button
                        className="navigation-button" 
                        onClick={() => navigate(-1)}>
                            ← Back
                     </button>
                )
            }
            
        </div>

      <div className="header-logo-side">
        {currLocation !== "/" && (
          <button
            className="sign-out-button"
            onClick={handleSignOut}
          >
            Sign Out
          </button>
        )}

        <img
          className="mcmaster-logo"
          src={mac_logo}
          alt="McMaster University Logo"
        />
      </div>
    </header>
  );
}