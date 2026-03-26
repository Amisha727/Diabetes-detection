import { useState } from 'react';
import PredictionResult from '../components/PredictionResult';
import ShapExplanation from '../components/ShapExplanation';
import { getCurrentUser, predictDiabetes } from '../services/api';

const FIELDS = [
    { name: 'pregnancies', label: 'Pregnancies', placeholder: 'e.g. 2', min: 0, max: 20, step: 1, info: 'Number of times pregnant' },
    { name: 'glucose', label: 'Glucose (mg/dL)', placeholder: 'e.g. 120', min: 0, max: 300, step: 1, info: 'Plasma glucose concentration (2h oral glucose tolerance test)' },
    { name: 'blood_pressure', label: 'Blood Pressure (mm Hg)', placeholder: 'e.g. 70', min: 0, max: 200, step: 1, info: 'Diastolic blood pressure' },
    { name: 'skin_thickness', label: 'Skin Thickness (mm)', placeholder: 'e.g. 20', min: 0, max: 100, step: 1, info: 'Triceps skin fold thickness' },
    { name: 'insulin', label: 'Insulin (mu U/ml)', placeholder: 'e.g. 80', min: 0, max: 900, step: 1, info: '2-Hour serum insulin' },
    { name: 'bmi', label: 'BMI', placeholder: 'e.g. 25.0', min: 0, max: 70, step: 0.1, info: 'Body mass index (weight/height²)' },
    { name: 'diabetes_pedigree', label: 'Diabetes Pedigree', placeholder: 'e.g. 0.5', min: 0, max: 3, step: 0.01, info: 'Diabetes pedigree function (genetic risk score)' },
    { name: 'age', label: 'Age (years)', placeholder: 'e.g. 33', min: 1, max: 120, step: 1, info: 'Age in years' },
];

const INITIAL = Object.fromEntries(FIELDS.map(f => [f.name, '']));

export default function PatientForm() {
    const [form, setForm] = useState(INITIAL);
    const [result, setResult] = useState(null);
    const [explanation, setExplanation] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const user = getCurrentUser();

    const handleChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setResult(null);
        setExplanation(null);

        // Validate all fields filled
        for (const f of FIELDS) {
            if (form[f.name] === '' || form[f.name] === undefined) {
                setError(`Please fill in ${f.label}`);
                return;
            }
        }

        const payload = {
            ...Object.fromEntries(
                Object.entries(form).map(([k, v]) => [k, parseFloat(v)])
            ),
        };

        setLoading(true);
        try {
            const pred = await predictDiabetes(payload);
            setResult(pred);
            setExplanation(pred.local_explanation || null);
        } catch (err) {
            setError(err.response?.data?.detail || 'Prediction failed. Is the backend running?');
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setForm(INITIAL);
        setResult(null);
        setExplanation(null);
        setError('');
    };

    const fillSample = () => {
        setForm({
            pregnancies: '6',
            glucose: '148',
            blood_pressure: '72',
            skin_thickness: '35',
            insulin: '0',
            bmi: '33.6',
            diabetes_pedigree: '0.627',
            age: '50',
        });
    };

    return (
        <div className="space-y-8">
            {/* Form Card */}
            <div className="bg-white rounded-2xl shadow-sm border p-6">
                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h2 className="text-2xl font-bold text-gray-800">Patient Assessment</h2>
                        <p className="text-gray-500 text-sm mt-1">Enter clinical variables to predict diabetes risk</p>
                    </div>
                    <button
                        type="button"
                        onClick={fillSample}
                        className="text-xs px-3 py-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-600 transition"
                    >
                        Fill Sample Data
                    </button>
                </div>

                <form onSubmit={handleSubmit}>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        {FIELDS.map(f => (
                            <div key={f.name} className="group">
                                <label className="block text-xs font-semibold text-gray-600 mb-1.5">
                                    {f.label}
                                </label>
                                <input
                                    type="number"
                                    name={f.name}
                                    value={form[f.name]}
                                    onChange={handleChange}
                                    placeholder={f.placeholder}
                                    min={f.min}
                                    max={f.max}
                                    step={f.step}
                                    className="w-full px-3 py-2.5 border border-gray-200 rounded-lg text-sm
                    focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition
                    placeholder:text-gray-300"
                                />
                                <p className="text-[10px] text-gray-400 mt-0.5">{f.info}</p>
                            </div>
                        ))}
                    </div>

                    <div className="mt-6 rounded-lg border border-blue-100 bg-blue-50 p-3 text-xs text-blue-700">
                        Active model: <strong>TabNet</strong> with SMOTE + Stratified K-Fold pipeline.
                        {!user && ' Sign in from Profile to save this prediction to your history.'}
                    </div>

                    {error && (
                        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
                            {error}
                        </div>
                    )}

                    <div className="mt-6 flex gap-3">
                        <button
                            type="submit"
                            disabled={loading}
                            className="px-6 py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium
                text-sm transition disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                        >
                            {loading ? (
                                <span className="flex items-center gap-2">
                                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" />
                                    </svg>
                                    Analyzing...
                                </span>
                            ) : 'Predict Diabetes Risk'}
                        </button>
                        <button
                            type="button"
                            onClick={handleReset}
                            className="px-5 py-2.5 border border-gray-200 hover:bg-gray-50 rounded-lg text-sm
                font-medium text-gray-600 transition"
                        >
                            Reset
                        </button>
                    </div>
                </form>
            </div>

            {/* Results */}
            {result && <PredictionResult result={result} />}
            {explanation && <ShapExplanation explanation={explanation} />}
        </div>
    );
}
