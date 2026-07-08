import React from "react";
import {
  Grid,
  Card,
  CardMedia,
  CardContent,
  CardActions,
  Typography,
  Button,
  Box,
  Chip,
} from "@mui/material";
import { TrendingUp, OpenInNew } from "@mui/icons-material";

const Recommendations = ({ movies }) => {
  if (!movies || movies.length === 0) {
    return (
      <Box sx={{ textAlign: "center", py: 10 }}>
        <Typography variant="h6" color="text.secondary">
          No recommendations found
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" sx={{ fontWeight: 700, mb: 1, color: "#1a1a1a" }}>
          Your recommendations
        </Typography>
        <Typography variant="body2" sx={{ color: "#6b7280" }}>
          Based on your ratings and viewing patterns
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {movies.map((movie, index) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={movie.movieId}>
            <Card
              elevation={0}
              sx={{
                height: 480,
                borderRadius: 3,
                border: "1px solid #e5e7eb",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
                transition: "all 0.25s",
                "&:hover": {
                  transform: "translateY(-6px)",
                  boxShadow: "0 12px 28px rgba(0,0,0,0.15)",
                },
              }}
            >
              <Box sx={{ position: "relative" }}>
                <CardMedia
                  component="img"
                  height="280"
                  image={movie.posterUrl || "https://via.placeholder.com/300x450?text=No+Poster"}
                  alt={movie.title}
                  sx={{ objectFit: "cover" }}
                />
                <Chip
                  label={`#${index + 1}`}
                  size="small"
                  sx={{
                    position: "absolute",
                    top: 12,
                    left: 12,
                    bgcolor: "#1a1a1a",
                    color: "white",
                    fontWeight: 700,
                    fontSize: "0.75rem",
                  }}
                />
              </Box>

              <CardContent sx={{ flexGrow: 1, p: 2.5 }}>
                <Typography
                  variant="h6"
                  sx={{
                    fontSize: "1rem",
                    fontWeight: 600,
                    mb: 0.5,
                    color: "#1a1a1a",
                    lineHeight: 1.3,
                    display: "-webkit-box",
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical",
                    overflow: "hidden",
                  }}
                >
                  {movie.title}
                </Typography>

                <Typography
                  variant="body2"
                  sx={{ color: "#6b7280", mb: 2, fontSize: "0.875rem" }}
                >
                  {movie.genres}
                </Typography>

                {movie.predicted_rating && (
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 1 }}>
                    <TrendingUp sx={{ fontSize: 18, color: "#10b981" }} />
                    <Typography
                      variant="body2"
                      sx={{ fontWeight: 600, color: "#10b981" }}
                    >
                      {movie.predicted_rating.toFixed(1)}/5.0
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{ color: "#9ca3af", ml: 0.5 }}
                    >
                      predicted
                    </Typography>
                  </Box>
                )}

                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Box
                    sx={{
                      height: 6,
                      flex: 1,
                      bgcolor: "#e5e7eb",
                      borderRadius: 1,
                      overflow: "hidden",
                    }}
                  >
                    <Box
                      sx={{
                        height: "100%",
                        width: `${movie.score * 100}%`,
                        bgcolor: "#5046e5",
                        transition: "width 0.3s",
                      }}
                    />
                  </Box>
                  <Typography
                    variant="caption"
                    sx={{
                      color: "#6b7280",
                      fontWeight: 600,
                      minWidth: 40,
                    }}
                  >
                    {(movie.score * 100).toFixed(0)}%
                  </Typography>
                </Box>
              </CardContent>

              <CardActions sx={{ p: 2.5, pt: 0 }}>
                <Button
                  size="small"
                  fullWidth
                  endIcon={<OpenInNew sx={{ fontSize: 16 }} />}
                  href={`https://www.google.com/search?q=${encodeURIComponent(
                    movie.title + " movie"
                  )}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  sx={{
                    textTransform: "none",
                    fontWeight: 600,
                    color: "#5046e5",
                    borderRadius: 1.5,
                    py: 1,
                    "&:hover": { bgcolor: "#f0f0ff" },
                  }}
                >
                  Learn more
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default Recommendations;
