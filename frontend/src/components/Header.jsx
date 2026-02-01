import "./Header.css";
import mac_logo from "./mcmaster-logo.png"
import { useLocation, useNavigate} from "react-router-dom"

export default function Header(){

    const location = useLocation();
    const navigate = useNavigate();

    const currLocation = location.pathname 

    return (
       <header className="Reading4All-header">

        <div classname="header-navigation">
        { 
            // only the upload pg should have a header to go to history pg
            currLocation==="/upload" && (
                <button
                    className="history-page-navigation-button" 
                    onClick={() => navigate("/session-history")}>
                        View History
                </button>
            )
        }
        </div>

        <div className="header-logo-side">
            <img
            className="mcmaster-logo"
            src={mac_logo}
            alt="McMaster University Logo"
            />
        </div>
      
       </header>
    );
}
