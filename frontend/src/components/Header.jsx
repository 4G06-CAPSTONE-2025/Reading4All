import "./Header.css";
import mac_logo from "./mcmaster-logo.png"

export default function Header(){

    return (
       <header className="Reading4All-header">
        <img
        className="mcmaster-logo"
        src={mac_logo}
        alt="McMaster University Logo"
        />
       </header>
    );
}
