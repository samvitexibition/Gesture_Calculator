# pyrefly: ignore [missing-import]
import pickle

try:
    with open('models/sign_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print('Model loaded successfully')
except Exception as e:
    print(f'Error loading model: {e}')
    import traceback
    traceback.print_exc()
