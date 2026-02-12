import { HashRouter as Router, Routes, Route } from "react-router-dom";

import Header from "./components/Header";
import HomeScreen from "./pages/HomeScreen/homeScreen";
import Login from "./pages/LogInScreen/login";
import SignUpScreen from "./pages/SignUpScreen/SignUpScreen";
import UploadScreen from "./pages/UploadScreen/UploadScreen";
import HistoryPage from "./pages/HistoryScreen/HistoryPage";

function App() {
  return (
    <Router>
      <Header />

      <Routes>
        {/* DEFAULT LANDING PAGE */}
        <Route path="/" element={<HomeScreen />} />

        {/* Auth */}
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<SignUpScreen />} />

        {/* App */}
        <Route path="/upload" element={<UploadScreen />} />
        <Route path="/session-history" element={<HistoryPage />} />
      </Routes>
    </Router>
  );
}

export default App;
