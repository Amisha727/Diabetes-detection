
const RISK_COLORS = {
    Low: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-700', badge: 'bg-green-100 text-green-800' },
    Medium: { bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-700', badge: 'bg-yellow-100 text-yellow-800' },
    High: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', badge: 'bg-red-100 text-red-800' },
};

export default function PredictionResult({ result }) {
    const colors = RISK_COLORS[result.risk_category] || RISK_COLORS.Medium;
    const pct = (result.probability * 100).toFixed(1);

    return (
        <div className={`rounded-2xl border-2 ${colors.border} ${colors.bg} p-6`}>
            <h3 className="text-lg font-bold text-gray-800 mb-4">Prediction Result</h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* Prediction */}
                <StatCard
                    label="Prediction"
                    value={result.prediction}
                    className={result.prediction === 'Diabetic' ? 'text-red-600' : 'text-green-600'}
                />

                {/* Probability */}
                <div className="bg-white rounded-xl p-4 shadow-sm">
                    <p className="text-xs text-gray-500 font-medium mb-2">Probability</p>
                    <div className="flex items-end gap-2">
                        <span className="text-2xl font-bold text-gray-800">{pct}%</span>
                    </div>
                    <div className="mt-2 h-2 bg-gray-100 rounded-full overflow-hidden">
                        <div
                            className={`h-full rounded-full transition-all duration-500 ${result.probability >= 0.5 ? 'bg-red-500' : 'bg-green-500'
                                }`}
                            style={{ width: `${pct}%` }}
                        />
                    </div>
                </div>

                {/* Risk Category */}
                <div className="bg-white rounded-xl p-4 shadow-sm">
                    <p className="text-xs text-gray-500 font-medium mb-2">Risk Category</p>
                    <span className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${colors.badge}`}>
                        {result.risk_category} Risk
                    </span>
                </div>

                {/* Confidence */}
                <StatCard label="Confidence" value={result.confidence} />

                {/* Model */}
                <StatCard
                    label="Model Used"
                    value={(result.model_used || 'tabnet').toUpperCase()}
                />

                {/* Prediction ID */}
                <StatCard
                    label="Prediction ID"
                    value={result.prediction_id ?? 'Not saved'}
                />
            </div>

            {Array.isArray(result.lifestyle_recommendations) && result.lifestyle_recommendations.length > 0 && (
                <div className="mt-5 rounded-xl bg-white p-4 shadow-sm">
                    <p className="text-sm font-semibold text-gray-700 mb-2">Lifestyle Recommendations</p>
                    <ul className="list-disc pl-5 text-sm text-gray-600 space-y-1">
                        {result.lifestyle_recommendations.map((item, idx) => (
                            <li key={idx}>{item}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}

function StatCard({ label, value, className = 'text-gray-800' }) {
    return (
        <div className="bg-white rounded-xl p-4 shadow-sm">
            <p className="text-xs text-gray-500 font-medium mb-1">{label}</p>
            <p className={`text-xl font-bold ${className}`}>{value}</p>
        </div>
    );
}
