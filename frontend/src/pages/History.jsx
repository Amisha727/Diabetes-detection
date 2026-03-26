import {
    CategoryScale,
    Chart as ChartJS,
    Legend,
    LinearScale,
    LineElement,
    PointElement,
    Title,
    Tooltip,
} from 'chart.js';
import { useEffect, useMemo, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { getHistory, isLoggedIn } from '../services/api';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const RISK_CLASSES = {
    Low: 'text-green-700 bg-green-50 border-green-200',
    Medium: 'text-yellow-700 bg-yellow-50 border-yellow-200',
    High: 'text-red-700 bg-red-50 border-red-200',
};

export default function History() {
    const [historyData, setHistoryData] = useState([]);
    const [trend, setTrend] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        async function loadHistory() {
            if (!isLoggedIn()) {
                setError('Please sign in from Profile to view patient history.');
                setLoading(false);
                return;
            }

            try {
                const res = await getHistory();
                setHistoryData(res.history || []);
                setTrend(res.trend || []);
            } catch (err) {
                setError(err.response?.data?.detail || 'Unable to load history.');
            } finally {
                setLoading(false);
            }
        }

        loadHistory();
    }, []);

    const chartData = useMemo(() => {
        return {
            labels: trend.map((p) => new Date(p.timestamp).toLocaleDateString()),
            datasets: [
                {
                    label: 'Diabetes Probability Trend',
                    data: trend.map((p) => Number((p.probability * 100).toFixed(2))),
                    borderColor: 'rgb(37, 99, 235)',
                    backgroundColor: 'rgba(37, 99, 235, 0.15)',
                    fill: true,
                    tension: 0.35,
                    pointRadius: 3,
                },
            ],
        };
    }, [trend]);

    if (loading) {
        return <div className="text-gray-600">Loading history...</div>;
    }

    if (error) {
        return <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-700">{error}</div>;
    }

    return (
        <div className="space-y-6">
            <div className="bg-white rounded-2xl border shadow-sm p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-1">Prediction History</h2>
                <p className="text-sm text-gray-500">Stored patient predictions and longitudinal risk trend.</p>
            </div>

            <div className="bg-white rounded-2xl border shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Risk Trend Chart</h3>
                {trend.length === 0 ? (
                    <p className="text-gray-500 text-sm">No trend data yet. Run predictions after signing in.</p>
                ) : (
                    <div className="h-[320px]">
                        <Line
                            data={chartData}
                            options={{
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {
                                    y: {
                                        min: 0,
                                        max: 100,
                                        title: { display: true, text: 'Probability (%)' },
                                    },
                                    x: {
                                        title: { display: true, text: 'Prediction Date' },
                                    },
                                },
                                plugins: {
                                    legend: { position: 'bottom' },
                                },
                            }}
                        />
                    </div>
                )}
            </div>

            <div className="bg-white rounded-2xl border shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Past Predictions</h3>
                {historyData.length === 0 ? (
                    <p className="text-gray-500 text-sm">No predictions saved yet.</p>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b text-left text-gray-500">
                                    <th className="pb-3">Date</th>
                                    <th className="pb-3">Prediction</th>
                                    <th className="pb-3">Probability</th>
                                    <th className="pb-3">Risk</th>
                                    <th className="pb-3">Model</th>
                                </tr>
                            </thead>
                            <tbody>
                                {historyData.map((item) => (
                                    <tr key={item.id} className="border-b border-gray-50">
                                        <td className="py-3 text-gray-700">{new Date(item.created_at).toLocaleString()}</td>
                                        <td className="py-3 font-medium text-gray-800">{item.prediction}</td>
                                        <td className="py-3 text-gray-700">{(item.probability * 100).toFixed(2)}%</td>
                                        <td className="py-3">
                                            <span className={`inline-block rounded-full border px-2 py-1 text-xs font-medium ${RISK_CLASSES[item.risk_category] || RISK_CLASSES.Medium}`}>
                                                {item.risk_category}
                                            </span>
                                        </td>
                                        <td className="py-3 uppercase text-gray-600">{item.model_used}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
}
