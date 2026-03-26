
export default function About() {
    return (
        <div className="max-w-3xl mx-auto space-y-8">
            <div className="bg-white rounded-2xl shadow-sm border p-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-4">About This Project</h2>
                <p className="text-gray-600 leading-relaxed mb-4">
                    This application implements the research paper{' '}
                    <strong>"AI Driven Approaches for Diabetes Detection in Healthcare"</strong>.
                    It provides a clinical decision support system that predicts diabetes risk and
                    explains the prediction using state-of-the-art explainable AI techniques.
                </p>

                <h3 className="text-lg font-semibold text-gray-800 mt-6 mb-3">Methodology</h3>
                <ul className="space-y-2 text-gray-600">
                    <li className="flex gap-2">
                        <span className="text-primary-500 font-bold">•</span>
                        <span><strong>Dataset:</strong> Pima Indians Diabetes Dataset (UCI ML Repository) — 768 samples, 8 features</span>
                    </li>
                    <li className="flex gap-2">
                        <span className="text-primary-500 font-bold">•</span>
                        <span><strong>Preprocessing:</strong> Median imputation for zero values, StandardScaler normalization</span>
                    </li>
                    <li className="flex gap-2">
                        <span className="text-primary-500 font-bold">•</span>
                        <span><strong>Models:</strong> Random Forest Classifier and XGBoost Classifier</span>
                    </li>
                    <li className="flex gap-2">
                        <span className="text-primary-500 font-bold">•</span>
                        <span><strong>Evaluation:</strong> Accuracy, Precision, Recall, F1-Score, ROC-AUC</span>
                    </li>
                    <li className="flex gap-2">
                        <span className="text-primary-500 font-bold">•</span>
                        <span><strong>Explainability:</strong> SHAP (SHapley Additive exPlanations) for global and local feature importance</span>
                    </li>
                </ul>

                <h3 className="text-lg font-semibold text-gray-800 mt-6 mb-3">Clinical Features</h3>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b text-left text-gray-500">
                                <th className="pb-2 pr-4 font-medium">Feature</th>
                                <th className="pb-2 font-medium">Description</th>
                            </tr>
                        </thead>
                        <tbody className="text-gray-600">
                            {[
                                ['Pregnancies', 'Number of times pregnant'],
                                ['Glucose', 'Plasma glucose concentration (2h OGTT)'],
                                ['Blood Pressure', 'Diastolic blood pressure (mm Hg)'],
                                ['Skin Thickness', 'Triceps skin fold thickness (mm)'],
                                ['Insulin', '2-Hour serum insulin (mu U/ml)'],
                                ['BMI', 'Body mass index (weight/height²)'],
                                ['Diabetes Pedigree', 'Diabetes pedigree function (genetic score)'],
                                ['Age', 'Age in years'],
                            ].map(([feat, desc]) => (
                                <tr key={feat} className="border-b border-gray-50">
                                    <td className="py-2 pr-4 font-medium text-gray-700">{feat}</td>
                                    <td className="py-2">{desc}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                <h3 className="text-lg font-semibold text-gray-800 mt-6 mb-3">Technology Stack</h3>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <h4 className="font-medium text-gray-700 mb-1">Backend</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                            <li>• Python / FastAPI</li>
                            <li>• scikit-learn / XGBoost</li>
                            <li>• SHAP</li>
                            <li>• pandas / NumPy</li>
                        </ul>
                    </div>
                    <div>
                        <h4 className="font-medium text-gray-700 mb-1">Frontend</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                            <li>• React 18</li>
                            <li>• TailwindCSS</li>
                            <li>• Chart.js</li>
                            <li>• Axios</li>
                        </ul>
                    </div>
                </div>
            </div>

            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 text-sm text-yellow-800">
                <strong>Disclaimer:</strong> This tool is intended for educational and research purposes only.
                It is not a substitute for professional medical advice, diagnosis, or treatment.
                Always consult a qualified healthcare provider.
            </div>
        </div>
    );
}
