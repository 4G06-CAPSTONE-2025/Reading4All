import { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";

export default function RequireAuth({ children }) {
  const [loading, setLoading] = useState(true);
  const [authenticated, setAuthenticated] = useState(false);

  useEffect(() => {
    fetch("https://reading4all-backend.onrender.com/api/session/", {
      credentials: "include",
    })
      .then((res) => {
        if (!res.ok) throw new Error();
        setAuthenticated(true);
      })
      .catch(() => setAuthenticated(false))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return null;

  return authenticated ? children : <Navigate to="/login" replace />;
}