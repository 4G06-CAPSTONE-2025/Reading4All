import { HashRouter as Router, Routes, Route } from "react-router-dom";
import Login from "./pages/LogInScreen/login";
import HomeScreen from "./pages/HomeScreen/homeScreen";
import ShowHistory from "./pages/ShowHistoryScreen/showHistory";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/main" element={<HomeScreen />} />
        <Route path="/session-history" element={<ShowHistory />} />
      </Routes>
    </Router>
  );
}

export default App;
