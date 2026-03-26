import {
    BarElement,
    CategoryScale,
    Chart as ChartJS,
    Filler,
    Legend,
    LinearScale,
    LineElement,
    PointElement,
    Title,
    Tooltip,
} from 'chart.js';
import { useEffect, useState } from 'react';
import { Bar, Line } from 'react-chartjs-2';
import { getFeatureImportance, getMetrics, getRocData } from '../services/api';

ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, Title, Tooltip, Legend, Filler);

export default function Dashboard() {
    const [importance, setImportance] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const [roc, setRoc] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        async function load() {
            try {
                const [imp, met, rocData] = await Promise.all([
                    getFeatureImportance(),
                    getMetrics(),
                    getRocData(),
                ]);
                setImportance(imp);
                setMetrics(met);
                setRoc(rocData);
            } catch (e) {
                setError('Failed to load dashboard data. Is the backend running?');
            } finally {
                setLoading(false);
            }
        }
        load();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin h-8 w-8 border-4 border-primary-500 border-t-transparent rounded-full" />
            </div>
        );
    }

    if (error) {
        return <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-red-600">{error}</div>;
    }

    return (
        <div className="space-y-8">
            <div className="rounded-xl border border-blue-100 bg-blue-50 p-4 text-sm text-blue-700">
                Evaluation dashboard for the research pipeline: TabNet + SMOTE + Stratified K-Fold + Optuna + SHAP.
            </div>

            {/* Metrics Cards */}
            {metrics && <MetricsCards metrics={metrics.tabnet} />}

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Feature Importance */}
                {importance && (
                    <div className="bg-white rounded-2xl shadow-sm border p-6">
                        <h3 className="text-lg font-bold text-gray-800 mb-4">Global SHAP Feature Importance</h3>
                        <FeatureImportanceChart data={importance.tabnet_shap} />
                    </div>
                )}

                {/* ROC Curve */}
                {roc && (
                    <div className="bg-white rounded-2xl shadow-sm border p-6">
                        <h3 className="text-lg font-bold text-gray-800 mb-4">ROC Curve</h3>
                        <RocChart roc={roc.tabnet} auc={metrics?.tabnet?.roc_auc} />
                    </div>
                )}
            </div>

            {importance?.tabnet_attention && (
                <div className="bg-white rounded-2xl shadow-sm border p-6">
                    <h3 className="text-lg font-bold text-gray-800 mb-4">TabNet Attention Feature Importance</h3>
                    <FeatureImportanceChart data={importance.tabnet_attention} />
                </div>
            )}
        </div>
    );
}


function MetricsCards({ metrics }) {
    if (!metrics) return null;

    const cards = [
        { label: 'Accuracy', value: metrics.accuracy, color: 'text-blue-600' },
        { label: 'Precision', value: metrics.precision, color: 'text-purple-600' },
        { label: 'Recall', value: metrics.recall, color: 'text-orange-600' },
        { label: 'F1 Score', value: metrics.f1_score, color: 'text-green-600' },
        { label: 'ROC-AUC', value: metrics.roc_auc, color: 'text-red-600' },
    ];

    return (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
            {cards.map(c => (
                <div key={c.label} className="bg-white rounded-xl shadow-sm border p-4 text-center">
                    <p className="text-xs text-gray-500 font-medium">{c.label}</p>
                    <p className={`text-2xl font-bold mt-1 ${c.color}`}>{(c.value * 100).toFixed(1)}%</p>
                </div>
            ))}
        </div>
    );
}


function FeatureImportanceChart({ data }) {
    if (!data) return null;

    const sorted = Object.entries(data).sort((a, b) => b[1] - a[1]);
    const labels = sorted.map(([k]) => k);
    const values = sorted.map(([, v]) => v);

    const chartData = {
        labels,
        datasets: [
            {
                label: 'Mean |SHAP|',
                data: values,
                backgroundColor: 'rgba(59, 130, 246, 0.7)',
                borderColor: 'rgb(59, 130, 246)',
                borderWidth: 1,
                borderRadius: 4,
            },
        ],
    };

    const options = {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { title: { display: true, text: 'Mean |SHAP Value|' }, grid: { color: 'rgba(0,0,0,0.05)' } },
            y: { grid: { display: false } },
        },
    };

    return (
        <div className="h-[320px]">
            <Bar data={chartData} options={options} />
        </div>
    );
}


function RocChart({ roc, auc }) {
    if (!roc) return null;

    const data = {
        labels: roc.fpr.map(v => v.toFixed(3)),
        datasets: [
            {
                label: `TabNet (AUC = ${auc})`,
                data: roc.tpr,
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                borderWidth: 2,
            },
            {
                label: 'Random Baseline',
                data: roc.fpr,
                borderColor: 'rgba(156,163,175,0.5)',
                borderDash: [6, 4],
                pointRadius: 0,
                borderWidth: 1,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
            x: { title: { display: true, text: 'False Positive Rate' }, ticks: { maxTicksLimit: 8 } },
            y: { title: { display: true, text: 'True Positive Rate' }, min: 0, max: 1 },
        },
    };

    return (
        <div className="h-[320px]">
            <Line data={data} options={options} />
        </div>
    );
}
