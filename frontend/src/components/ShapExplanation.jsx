import {
    BarElement,
    CategoryScale,
    Chart as ChartJS,
    Legend,
    LinearScale,
    Title,
    Tooltip,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export default function ShapExplanation({ explanation }) {
    const { features, shap_values } = explanation;

    // Sort by absolute SHAP value
    const indexed = features.map((f, i) => ({ feature: f, value: shap_values[i] }));
    indexed.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

    const labels = indexed.map(d => d.feature);
    const values = indexed.map(d => d.value);
    const colors = values.map(v => (v >= 0 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(34, 197, 94, 0.8)'));
    const borderColors = values.map(v => (v >= 0 ? 'rgb(239, 68, 68)' : 'rgb(34, 197, 94)'));

    const data = {
        labels,
        datasets: [
            {
                label: 'SHAP Value (impact on prediction)',
                data: values,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4,
            },
        ],
    };

    const options = {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            title: {
                display: true,
                text: 'Patient-Specific SHAP Explanation',
                font: { size: 14, weight: 'bold' },
                padding: { bottom: 16 },
            },
            tooltip: {
                callbacks: {
                    label: (ctx) => {
                        const v = ctx.raw;
                        return `${v > 0 ? '↑ Increases' : '↓ Decreases'} diabetes risk by ${Math.abs(v).toFixed(4)}`;
                    },
                },
            },
        },
        scales: {
            x: {
                title: { display: true, text: 'SHAP Value', font: { weight: 'bold' } },
                grid: { color: 'rgba(0,0,0,0.05)' },
            },
            y: {
                grid: { display: false },
            },
        },
    };

    return (
        <div className="bg-white rounded-2xl shadow-sm border p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-2">Explainability — SHAP Analysis</h3>
            <p className="text-sm text-gray-500 mb-4">
                <span className="inline-block w-3 h-3 rounded bg-red-500 mr-1" /> Increases risk &nbsp;
                <span className="inline-block w-3 h-3 rounded bg-green-500 mr-1" /> Decreases risk
            </p>
            <div className="h-[360px]">
                <Bar data={data} options={options} />
            </div>

            {/* Table view */}
            <div className="mt-6 overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b text-left text-gray-500">
                            <th className="pb-2 font-medium">Feature</th>
                            <th className="pb-2 font-medium text-right">SHAP Value</th>
                            <th className="pb-2 font-medium text-right">Direction</th>
                        </tr>
                    </thead>
                    <tbody>
                        {indexed.map((d, i) => (
                            <tr key={i} className="border-b border-gray-50 hover:bg-gray-50">
                                <td className="py-2 font-medium text-gray-700">{d.feature}</td>
                                <td className="py-2 text-right font-mono text-xs">{d.value.toFixed(6)}</td>
                                <td className={`py-2 text-right font-medium ${d.value >= 0 ? 'text-red-600' : 'text-green-600'}`}>
                                    {d.value >= 0 ? '↑ Risk' : '↓ Protective'}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
