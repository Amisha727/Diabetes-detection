import { useEffect, useState } from 'react';
import {
    clearAuthSession,
    getCurrentUser,
    getProfile,
    isLoggedIn,
    login,
    register,
    setAuthSession,
} from '../services/api';

const LOGIN_INITIAL = { username: '', password: '' };
const REGISTER_INITIAL = { username: '', email: '', password: '', full_name: '' };

export default function Profile() {
    const [user, setUser] = useState(getCurrentUser());
    const [loginForm, setLoginForm] = useState(LOGIN_INITIAL);
    const [registerForm, setRegisterForm] = useState(REGISTER_INITIAL);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [message, setMessage] = useState('');

    useEffect(() => {
        async function refreshProfile() {
            if (!isLoggedIn()) return;
            try {
                const profile = await getProfile();
                setUser(profile);
            } catch {
                clearAuthSession();
                setUser(null);
            }
        }
        refreshProfile();
    }, []);

    const handleLogin = async (e) => {
        e.preventDefault();
        setError('');
        setMessage('');
        setLoading(true);
        try {
            const res = await login(loginForm);
            setAuthSession(res.access_token, res.user);
            setUser(res.user);
            setMessage('Login successful. Your predictions will now be stored in history.');
            setLoginForm(LOGIN_INITIAL);
        } catch (err) {
            setError(err.response?.data?.detail || 'Login failed.');
        } finally {
            setLoading(false);
        }
    };

    const handleRegister = async (e) => {
        e.preventDefault();
        setError('');
        setMessage('');
        setLoading(true);
        try {
            const res = await register(registerForm);
            setAuthSession(res.access_token, res.user);
            setUser(res.user);
            setMessage('Registration successful. You are now signed in.');
            setRegisterForm(REGISTER_INITIAL);
        } catch (err) {
            setError(err.response?.data?.detail || 'Registration failed.');
        } finally {
            setLoading(false);
        }
    };

    const handleLogout = () => {
        clearAuthSession();
        setUser(null);
        setMessage('Logged out successfully.');
        setError('');
    };

    return (
        <div className="space-y-6 max-w-4xl mx-auto">
            <div className="bg-white rounded-2xl border shadow-sm p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-1">Profile & Authentication</h2>
                <p className="text-sm text-gray-500">Register or login to enable stored prediction history per user.</p>
            </div>

            {message && <div className="rounded-lg border border-green-200 bg-green-50 p-3 text-sm text-green-700">{message}</div>}
            {error && <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div>}

            {user ? (
                <div className="bg-white rounded-2xl border shadow-sm p-6 space-y-4">
                    <h3 className="text-lg font-semibold text-gray-800">Account Information</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <Info label="Username" value={user.username} />
                        <Info label="Email" value={user.email} />
                        <Info label="Full Name" value={user.full_name || '-'} />
                        <Info label="User ID" value={String(user.id)} />
                    </div>
                    <button
                        onClick={handleLogout}
                        className="px-4 py-2 rounded-lg bg-gray-800 text-white hover:bg-black transition"
                    >
                        Logout
                    </button>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <form onSubmit={handleLogin} className="bg-white rounded-2xl border shadow-sm p-6 space-y-4">
                        <h3 className="text-lg font-semibold text-gray-800">Login</h3>
                        <Input label="Username" value={loginForm.username} onChange={(v) => setLoginForm({ ...loginForm, username: v })} />
                        <Input label="Password" type="password" value={loginForm.password} onChange={(v) => setLoginForm({ ...loginForm, password: v })} />
                        <button disabled={loading} className="w-full px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition disabled:opacity-60">
                            {loading ? 'Signing in...' : 'Login'}
                        </button>
                    </form>

                    <form onSubmit={handleRegister} className="bg-white rounded-2xl border shadow-sm p-6 space-y-4">
                        <h3 className="text-lg font-semibold text-gray-800">Register</h3>
                        <Input label="Username" value={registerForm.username} onChange={(v) => setRegisterForm({ ...registerForm, username: v })} />
                        <Input label="Email" type="email" value={registerForm.email} onChange={(v) => setRegisterForm({ ...registerForm, email: v })} />
                        <Input label="Full Name" value={registerForm.full_name} onChange={(v) => setRegisterForm({ ...registerForm, full_name: v })} />
                        <Input label="Password" type="password" value={registerForm.password} onChange={(v) => setRegisterForm({ ...registerForm, password: v })} />
                        <button disabled={loading} className="w-full px-4 py-2 rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 transition disabled:opacity-60">
                            {loading ? 'Creating account...' : 'Register'}
                        </button>
                    </form>
                </div>
            )}
        </div>
    );
}

function Input({ label, value, onChange, type = 'text' }) {
    return (
        <label className="block">
            <span className="block text-sm font-medium text-gray-600 mb-1">{label}</span>
            <input
                type={type}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
                required
            />
        </label>
    );
}

function Info({ label, value }) {
    return (
        <div className="rounded-lg border border-gray-100 bg-gray-50 p-3">
            <p className="text-xs text-gray-500">{label}</p>
            <p className="font-medium text-gray-800 mt-1">{value}</p>
        </div>
    );
}
