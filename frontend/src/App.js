import { HashRouter as Router, Routes, Route } from "react-router-dom";

import Header from "./components/Header";
import RequireAuth from "./pages/RequireAuth";
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
        <Route path="/" element={<HomeScreen />} />

        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<SignUpScreen />} />

        <Route
          path="/upload"
          element={
            <RequireAuth>
              <UploadScreen />
            </RequireAuth>
          }
        />

        <Route
          path="/session-history"
          element={
            <RequireAuth>
              <HistoryPage />
            </RequireAuth>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;