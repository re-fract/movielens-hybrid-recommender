const API_URL = "http://127.0.0.1:5000";

export const getRecommendations = async (userId, numRecommendations) => {
  const response = await fetch(`${API_URL}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ userId: Number(userId), numRecommendations: Number(numRecommendations), alpha: 0.7 }),
  });
  if (!response.ok) throw new Error("Network response was not ok");
  const data = await response.json();
  return data.recommendations;
};

export const fetchMovieSample = async () => {
  // TODO: Implement a backend endpoint for sample movies or hardcode some for demo
  const sampleMovies = [
    { movieId: 1, title: "Toy Story (1995)", genres: "Animation|Children|Comedy" },
    { movieId: 31, title: "Dangerous Minds (1995)", genres: "Drama" },
    { movieId: 50, title: "Star Wars: Episode IV - A New Hope (1977)", genres: "Action|Adventure|Sci-Fi" },
    { movieId: 150, title: "Apollo 13 (1995)", genres: "Adventure|Drama|Thriller" },
    { movieId: 163, title: "Dead Poets Society (1989)", genres: "Drama" },
    // Add more or create a real backend endpoint /movies/sample
  ];
  return sampleMovies;
};

export const getRecommendationsForNewUser = async (userSelections, numRecommendations) => {
  const response = await fetch(`${API_URL}/recommend-newuser`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ userSelections, numRecommendations: Number(numRecommendations), alpha: 0.7 }),
  });
  if (!response.ok) throw new Error("Network response was not ok");
  const data = await response.json();
  return data.recommendations;
};
