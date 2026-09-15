import { api, type ApiUser } from "@/lib/api";
import { useCallback, useEffect, useState } from "react";

type UseAuthOptions = {
  redirectOnUnauthenticated?: boolean;
  redirectPath?: string;
};

export function useAuth(options?: UseAuthOptions) {
  const { redirectOnUnauthenticated = false, redirectPath = "/" } = options ?? {};
  const [user, setUser] = useState<ApiUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const result = await api.auth.me();
      setUser(result.user);
      setError(null);
      localStorage.setItem("agrismart-user", JSON.stringify(result.user));
      if (redirectOnUnauthenticated && !result.user) window.location.href = redirectPath;
    } catch (nextError) {
      setError(nextError as Error);
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, [redirectOnUnauthenticated, redirectPath]);

  useEffect(() => { void refresh(); }, [refresh]);

  const logout = useCallback(async () => {
    await api.auth.logout();
    setUser(null);
    localStorage.removeItem("agrismart-user");
  }, []);

  return { user, loading, error, isAuthenticated: Boolean(user), refresh, logout };
}
