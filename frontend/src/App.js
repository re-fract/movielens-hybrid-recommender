import React, { useState, useEffect } from "react";
import {
  Container,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Button,
  TextField,
  Alert,
  Chip,
  Stack,
} from "@mui/material";
import { LocalMovies, Star, AutoAwesome } from "@mui/icons-material";
import MovieSelector from "./components/MovieSelector";
import Recommendations from "./components/Recommendations";

const API_URL = "http://127.0.0.1:5001";

function App() {
  const [step, setStep] = useState("rate");
  const [sampleMovies, setSampleMovies] = useState([]);
  const [userRatings, setUserRatings] = useState({});
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [userId, setUserId] = useState("");
  const [numRec, setNumRec] = useState(15);
  const [error, setError] = useState("");
  const [loadingSample, setLoadingSample] = useState(true);

  useEffect(() => {
    fetchSampleMovies();
  }, []);

  const fetchSampleMovies = async () => {
    setLoadingSample(true);
    setError("");
    try {
      const res = await fetch(`${API_URL}/movies-sample`);
      if (!res.ok) throw new Error("Failed to fetch movies");
      const data = await res.json();
      setSampleMovies(data);
    } catch (err) {
      setError("Couldn't connect to the server. Make sure it's running on port 5001.");
      console.error(err);
    }
    setLoadingSample(false);
  };

  const handleMovieRate = (movieId, rating) => {
    setUserRatings((prev) => ({ ...prev, [movieId]: rating }));
  };

  const handleRateSubmit = async () => {
    setLoading(true);
    setError("");

    try {
      const selections = Object.entries(userRatings).map(([movieId, rating]) => ({
        movieId: Number(movieId),
        rating,
      }));

      const resp = await fetch(`${API_URL}/recommend-newuser`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userSelections: selections,
          numRecommendations: numRec,
          alpha: 0.6,
          beta: 0.4,
        }),
      });

      if (!resp.ok) {
        const errorData = await resp.json();
        throw new Error(errorData.error || "Failed to get recommendations");
      }

      const data = await resp.json();
      setRecommendations(data.recommendations);
      setStep("recommend");
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
      console.error(err);
    }
    setLoading(false);
  };

  const handleExistingSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const resp = await fetch(`${API_URL}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: parseInt(userId),
          numRecommendations: numRec,
          alpha: 0.6,
          beta: 0.3,
          gamma: 0.1,
        }),
      });

      if (!resp.ok) {
        const data = await resp.json();
        throw new Error(data.error || "Failed to get recommendations");
      }

      const data = await resp.json();
      setRecommendations(data.recommendations);
      setStep("recommend");
    } catch (err) {
      setError(err.message || "Couldn't find that user ID.");
      console.error(err);
    }
    setLoading(false);
  };

  const resetApp = () => {
    setStep("rate");
    setUserRatings({});
    setRecommendations([]);
    setError("");
    setUserId("");
    fetchSampleMovies();
  };

  const ratedCount = Object.keys(userRatings).length;
  const canSubmit = ratedCount >= 15;

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "#fafbfc", py: 4 }}>
      <Container maxWidth="lg">
        {/* Header */}
        <Box sx={{ textAlign: "center", mb: 5 }}>
          <LocalMovies sx={{ fontSize: 50, color: "#5046e5", mb: 1 }} />
          <Typography variant="h3" sx={{ fontWeight: 700, color: "#1a1a1a", mb: 1 }}>
            Movie Recommender
          </Typography>
          <Typography variant="body1" sx={{ color: "#6b7280" }}>
            Get personalized movie picks based on your taste
          </Typography>
        </Box>

        {/* Main Content */}
        <Paper
          elevation={0}
          sx={{
            p: { xs: 3, md: 5 },
            borderRadius: 3,
            border: "1px solid #e5e7eb",
            bgcolor: "#ffffff",
          }}
        >
          {error && (
            <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError("")}>
              {error}
            </Alert>
          )}

          {/* NEW USER - RATE MOVIES */}
          {step === "rate" && (
            <>
              <Box sx={{ textAlign: "center", mb: 4 }}>
                <Typography variant="h5" sx={{ fontWeight: 600, mb: 1.5, color: "#1a1a1a" }}>
                  Rate some movies
                </Typography>
                <Typography variant="body2" sx={{ color: "#6b7280", mb: 3 }}>
                  We need at least 15 ratings to understand your preferences
                </Typography>
                <Stack direction="row" spacing={2} justifyContent="center">
                  <Chip
                    label={`${ratedCount} / 15`}
                    color={canSubmit ? "success" : "default"}
                    sx={{ fontWeight: 600 }}
                  />
                </Stack>
              </Box>

              {loadingSample ? (
                <Box sx={{ textAlign: "center", py: 10 }}>
                  <CircularProgress size={50} sx={{ color: "#5046e5" }} />
                  <Typography sx={{ mt: 2, color: "#6b7280" }}>Loading movies...</Typography>
                </Box>
              ) : (
                <>
                  <MovieSelector
                    movies={sampleMovies}
                    userRatings={userRatings}
                    onRate={handleMovieRate}
                  />

                  <Box sx={{ mt: 5, display: "flex", justifyContent: "center", gap: 2, flexWrap: "wrap" }}>
                    <Button
                      variant="contained"
                      size="large"
                      onClick={handleRateSubmit}
                      disabled={!canSubmit || loading}
                      startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <AutoAwesome />}
                      sx={{
                        px: 4,
                        py: 1.5,
                        bgcolor: "#5046e5",
                        textTransform: "none",
                        fontSize: "1rem",
                        fontWeight: 600,
                        borderRadius: 2,
                        "&:hover": { bgcolor: "#3d35c9" },
                        "&:disabled": { bgcolor: "#d1d5db" },
                      }}
                    >
                      {loading ? "Getting recommendations..." : "Show me recommendations"}
                    </Button>
                    <Button
                      variant="outlined"
                      size="large"
                      onClick={() => setStep("existing")}
                      sx={{
                        px: 4,
                        py: 1.5,
                        textTransform: "none",
                        fontSize: "1rem",
                        fontWeight: 600,
                        borderRadius: 2,
                        borderColor: "#d1d5db",
                        color: "#4b5563",
                        "&:hover": { borderColor: "#9ca3af", bgcolor: "#f9fafb" },
                      }}
                    >
                      I'm an existing user
                    </Button>
                  </Box>

                  {!canSubmit && ratedCount > 0 && (
                    <Typography variant="body2" sx={{ textAlign: "center", mt: 3, color: "#9ca3af" }}>
                      {15 - ratedCount} more to go
                    </Typography>
                  )}
                </>
              )}
            </>
          )}

          {/* EXISTING USER */}
          {step === "existing" && (
            <Box sx={{ maxWidth: 480, mx: "auto" }}>
              <Typography variant="h5" sx={{ fontWeight: 600, mb: 3, textAlign: "center", color: "#1a1a1a" }}>
                Welcome back
              </Typography>

              <form onSubmit={handleExistingSubmit}>
                <TextField
                  label="User ID"
                  type="number"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  fullWidth
                  required
                  helperText="Enter your user ID (e.g. 1, 2, 3...)"
                  inputProps={{ min: 1 }}
                  sx={{ mb: 3 }}
                />
                <TextField
                  label="Number of recommendations"
                  type="number"
                  value={numRec}
                  onChange={(e) => setNumRec(parseInt(e.target.value))}
                  fullWidth
                  inputProps={{ min: 5, max: 50 }}
                  sx={{ mb: 4 }}
                />
                <Button
                  type="submit"
                  variant="contained"
                  fullWidth
                  size="large"
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <AutoAwesome />}
                  sx={{
                    py: 1.5,
                    bgcolor: "#5046e5",
                    textTransform: "none",
                    fontSize: "1rem",
                    fontWeight: 600,
                    borderRadius: 2,
                    "&:hover": { bgcolor: "#3d35c9" },
                  }}
                >
                  {loading ? "Loading..." : "Get recommendations"}
                </Button>
              </form>

              <Button
                fullWidth
                onClick={() => setStep("rate")}
                sx={{
                  mt: 2,
                  textTransform: "none",
                  color: "#6b7280",
                  "&:hover": { bgcolor: "#f9fafb" },
                }}
              >
                ← Back to rating
              </Button>
            </Box>
          )}

          {/* RECOMMENDATIONS */}
          {step === "recommend" && (
            <>
              <Recommendations movies={recommendations} />
              <Box sx={{ mt: 5, textAlign: "center" }}>
                <Button
                  variant="outlined"
                  size="large"
                  onClick={resetApp}
                  sx={{
                    px: 4,
                    textTransform: "none",
                    fontWeight: 600,
                    borderColor: "#d1d5db",
                    color: "#4b5563",
                    "&:hover": { borderColor: "#9ca3af", bgcolor: "#f9fafb" },
                  }}
                >
                  Start over
                </Button>
              </Box>
            </>
          )}
        </Paper>

        {/* Footer */}
        <Typography variant="caption" sx={{ display: "block", textAlign: "center", mt: 4, color: "#9ca3af" }}>
          MovieLens dataset • SVD algorithm • Flask + React
        </Typography>
      </Container>
    </Box>
  );
}

export default App;
