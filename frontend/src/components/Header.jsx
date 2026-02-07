import "./Header.css";
import mac_logo from "./mcmaster-logo.png"
import { useLocation, useNavigate} from "react-router-dom"

export default function Header(){

    const location = useLocation();
    const navigate = useNavigate();

    const currLocation = location.pathname 

    return (
       <header className="Reading4All-header">

        <div className="header-navigation">
        
             {/* only the upload pg should have a header to go to history pg  */}
            {
                currLocation==="/upload" && (
                <button
                    className="navigation-button" 
                    onClick={() => navigate("/session-history")}>
                        View History
                </button>
            )}


            {
                currLocation==="/session-history" && (
                    <button
                        className="navigation-button" 
                        onClick={() => navigate(-1)}>
                            ← Back
                     </button>
                )
            }
            
        </div>

        <div className="header-logo-side">

            {
                currLocation !=="/" && (
                    <button
                        className="sign-out-button" 
                        alt="McMaster University Logo"
                        onClick={() => navigate("/")}
                    >
                    Sign Out
                    </button>
                )
            }

            <img
            className="mcmaster-logo"
            src={mac_logo}
            alt="McMaster University Logo"
            />
        </div>
      
       </header>
    );
}
