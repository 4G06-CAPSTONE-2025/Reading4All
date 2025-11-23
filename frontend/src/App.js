import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Login from "./pages/LogInScreen/login";
import HomeScreen from "./pages/HomeScreen/homeScreen";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/main" element={<HomeScreen />} />
      </Routes>
    </Router>
  );
}

export default App;
