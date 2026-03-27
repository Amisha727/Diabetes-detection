import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';
const TOKEN_KEY = 'diabetes_token';
const USER_KEY = 'diabetes_user';

const api = axios.create({ baseURL: API_BASE });

api.interceptors.request.use((config) => {
    const token = localStorage.getItem(TOKEN_KEY);
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

export function setAuthSession(token, user) {
    if (token) {
        localStorage.setItem(TOKEN_KEY, token);
    }
    if (user) {
        localStorage.setItem(USER_KEY, JSON.stringify(user));
    }
}

export function clearAuthSession() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
}

export function getCurrentUser() {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
}

export function isLoggedIn() {
    return Boolean(localStorage.getItem(TOKEN_KEY));
}

export async function predictDiabetes(data) {
    const response = await api.post('/predict', data);
    return response.data;
}

export async function explainPrediction(predictionId) {
    const response = await api.get('/explain', {
        params: predictionId ? { prediction_id: predictionId } : {},
    });
    return response.data;
}

export async function getFeatureImportance() {
    const response = await api.get('/feature-importance');
    return response.data;
}

export async function getMetrics() {
    const response = await api.get('/metrics');
    return response.data;
}

export async function getRocData() {
    const response = await api.get('/roc');
    return response.data;
}

export async function getHistory() {
    const response = await api.get('/history');
    return response.data;
}

export async function getProfile() {
    const response = await api.get('/profile');
    return response.data;
}

export async function register(payload) {
    const response = await api.post('/register', payload);
    return response.data;
}

export async function login(payload) {
    const response = await api.post('/login', payload);
    return response.data;
}

export default api;
