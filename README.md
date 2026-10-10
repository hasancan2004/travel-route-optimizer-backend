# Travel Route Optimizer Backend

Backend service for the Travel Route Optimizer mobile application.

The project is built with FastAPI and handles route generation, place discovery, budget estimation, itinerary storage and AI-assisted trip input parsing.

## Features

- Route optimization based on user interests and budget
- Multi-day itinerary generation
- Place scoring and ranking
- Nearby place discovery with Google Places API
- City-based tourist spot retrieval
- Budget prediction with a machine learning model
- AI-based parsing of natural language trip requests
- Saving and sharing itineraries with Supabase
- Public itinerary discovery
- User-specific itinerary retrieval

## Tech Stack

- Python
- FastAPI
- Supabase
- PostgreSQL
- scikit-learn
- pandas
- NumPy
- Google Places API
- Gemini API
- Uvicorn

## Route Optimization

The route generation process consists of several steps.

### Place Scoring

Places are evaluated according to:

- User interests
- Rating
- Estimated entry cost
- Available budget

A nearest-neighbor based approach is used to rank places according to these features.

### Multi-Day Planning

Selected places are divided into multiple days using KMeans clustering based on geographic coordinates.

Within each day, locations are ordered using a nearest-location approach to reduce unnecessary travel distance.

## Budget Prediction

The backend includes a Random Forest regression model for estimating trip costs.

The prediction currently considers factors such as:

- Number of places
- Food-related locations
- Historical locations
- Shopping locations
- Selected city

The result is compared with the user's budget to provide a simple budget status.

## AI Integration

The API includes an endpoint for converting natural language trip requests into structured data.

For example, a request such as:

```text
3 günlük ucuz bir İstanbul turu çiz
