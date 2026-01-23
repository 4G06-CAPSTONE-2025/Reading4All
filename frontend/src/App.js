import { HashRouter as Router, Routes, Route } from "react-router-dom";
import Login from "./pages/LogInScreen/login";
import HomeScreen from "./pages/HomeScreen/homeScreen";
import ShowHistory from "./pages/ShowHistoryScreen/showHistory";
import HistoryPage from "./pages/HistoryScreen/HistoryPage";


function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/main" element={<HomeScreen />} />
        <Route path="/session-history" element={<HistoryPage />} />
      </Routes>
    </Router>
  );
}

export default App;
