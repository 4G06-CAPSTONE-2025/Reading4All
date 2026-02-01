import { HashRouter as Router, Routes, Route } from "react-router-dom";
import Login from "./pages/LogInScreen/login";
import HistoryPage from "./pages/HistoryScreen/HistoryPage";
import Header from "./components/Header";
import UploadScreen from "./pages/UploadScreen/UploadScreen"

function App() {
  return (
    <Router>
    <Header />
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/upload" element={<UploadScreen />} />
        <Route path="/session-history" element={<HistoryPage />} />
      </Routes>
      </Router>


  );
}

export default App;
