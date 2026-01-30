import { HashRouter as Router, Routes, Route } from "react-router-dom";
import Login from "./pages/LogInScreen/login";
import HomeScreen from "./pages/HomeScreen/homeScreen";
import HistoryPage from "./pages/HistoryScreen/HistoryPage";
import Header from "./components/Header";


function App() {
  return (
    <>
    <Header />
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/main" element={<HomeScreen />} />
        <Route path="/session-history" element={<HistoryPage />} />
      </Routes>
    </Router>
    </>

  );
}

export default App;
