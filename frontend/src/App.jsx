import { FiActivity, FiBarChart2, FiClock, FiUser } from 'react-icons/fi';
import { NavLink, Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import History from './pages/History';
import PatientForm from './pages/PatientForm';
import Profile from './pages/Profile';

export default function App() {
    return (
        <Router>
            <div className="min-h-screen flex flex-col">
                {/* Header */}
                <header className="bg-gradient-to-r from-primary-700 to-primary-900 text-white shadow-lg">
                    <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <span className="text-3xl">🩺</span>
                            <div>
                                <h1 className="text-xl font-bold leading-tight">Diabetes Detection</h1>
                                <p className="text-primary-200 text-xs">AI-Driven Clinical Decision Support</p>
                            </div>
                        </div>
                        <nav className="flex gap-1">
                            <NavItem to="/" icon={<FiActivity />} label="Predict" />
                            <NavItem to="/dashboard" icon={<FiBarChart2 />} label="Dashboard" />
                            <NavItem to="/history" icon={<FiClock />} label="History" />
                            <NavItem to="/profile" icon={<FiUser />} label="Profile" />
                        </nav>
                    </div>
                </header>

                {/* Content */}
                <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-8">
                    <Routes>
                        <Route path="/" element={<PatientForm />} />
                        <Route path="/dashboard" element={<Dashboard />} />
                        <Route path="/history" element={<History />} />
                        <Route path="/profile" element={<Profile />} />
                    </Routes>
                </main>

                {/* Footer */}
                <footer className="bg-gray-100 border-t text-center text-xs text-gray-500 py-3">
                    AI-Driven Diabetes Detection &mdash; Research Implementation &copy; 2026
                </footer>
            </div>
        </Router>
    );
}

function NavItem({ to, icon, label }) {
    return (
        <NavLink
            to={to}
            className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${isActive ? 'bg-white/20 text-white' : 'text-primary-200 hover:bg-white/10 hover:text-white'
                }`
            }
        >
            {icon}
            {label}
        </NavLink>
    );
}
