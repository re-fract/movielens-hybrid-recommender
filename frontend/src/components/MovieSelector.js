import React from "react";
import {
  Grid,
  Card,
  CardContent,
  CardMedia,
  Rating,
  Typography,
  Box,
} from "@mui/material";

const MovieSelector = ({ movies, userRatings, onRate }) => {
  if (!movies || movies.length === 0) {
    return (
      <Typography align="center" color="text.secondary">
        No movies available
      </Typography>
    );
  }

  const handleRatingChange = (movieId, value) => {
    // if rating cleared (value === null), remove it
    if (value === null) {
      const updatedRatings = { ...userRatings };
      delete updatedRatings[movieId];
      onRate(movieId, null, updatedRatings);
    } else {
      onRate(movieId, value);
    }
  };

  return (
    <Grid container spacing={2.5}>
      {movies.map((movie) => {
        const rating = userRatings[movie.movieId] || 0;
        const isRated = rating > 0;

        return (
          <Grid item xs={12} sm={6} md={4} lg={3} key={movie.movieId}>
            <Card
              elevation={0}
              sx={{
                height: 420,
                border: isRated ? "2px solid #5046e5" : "1px solid #e5e7eb",
                borderRadius: 3,
                bgcolor: isRated ? "#f0f0ff" : "#ffffff",
                transition: "all 0.25s",
                position: "relative",
                overflow: "hidden",
                cursor: "pointer",
                display: "flex",
                flexDirection: "column",
                "&:hover": {
                  borderColor: isRated ? "#5046e5" : "#d1d5db",
                  transform: "translateY(-4px)",
                  boxShadow: "0 8px 24px rgba(0,0,0,0.1)",
                },
              }}
            >
              {/* Poster */}
              <CardMedia
                component="img"
                image={
                  movie.posterUrl ||
                  "https://via.placeholder.com/300x450?text=No+Poster"
                }
                alt={movie.title}
                sx={{
                  height: 240,
                  objectFit: "cover",
                  borderBottom: "1px solid #e5e7eb",
                }}
              />

              {/* Content */}
              <CardContent
                sx={{
                  flexGrow: 1,
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  p: 2.5,
                }}
              >
                <Box>
                  <Typography
                    variant="subtitle1"
                    sx={{
                      fontWeight: 600,
                      mb: 0.5,
                      color: "#1a1a1a",
                      lineHeight: 1.3,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      display: "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical",
                    }}
                  >
                    {movie.title}
                  </Typography>

                  <Typography
                    variant="caption"
                    sx={{ color: "#6b7280", display: "block", mb: 1 }}
                  >
                    {movie.genres}
                  </Typography>
                </Box>

                <Box sx={{ textAlign: "center" }}>
                  <Rating
                    value={rating}
                    max={5}
                    size="medium"
                    onChange={(_, value) => handleRatingChange(movie.movieId, value)}
                    sx={{
                      "& .MuiRating-iconFilled": { color: "#f59e0b" },
                      "& .MuiRating-iconHover": { color: "#fbbf24" },
                    }}
                  />
                  {isRated && (
                    <Typography
                      variant="caption"
                      sx={{
                        display: "block",
                        mt: 0.5,
                        color: "#6b7280",
                        fontWeight: 500,
                      }}
                    >
                      You rated: {rating}/5
                    </Typography>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        );
      })}
    </Grid>
  );
};

export default MovieSelector;
