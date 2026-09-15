# Complete System Documentation - PashuCare

## Table of Contents
1. [System Overview](#system-overview)
2. [Database Architecture](#database-architecture)
3. [Application Architecture](#application-architecture)
4. [Complete Route Documentation](#complete-route-documentation)
5. [Frontend Architecture](#frontend-architecture)
6. [User Roles & Workflows](#user-roles--workflows)
7. [Feature Deep Dive](#feature-deep-dive)
8. [Technology Stack Details](#technology-stack-details)

---

## System Overview

**PashuCare** is a comprehensive, bilingual (English/Marathi), AI-powered livestock health management platform built with Flask, designed for farmers and veterinarians in India.

### Core Capabilities

1. **AI Disease Prediction**: Hierarchical two-stage ML system (85%+ accuracy)
2. **Farm Management**: Track animals, lands, vaccinations, and health records
3. **Veterinary Portal**: Professional tools for vets to manage patients
4. **Multilingual**: Full English/Marathi support
5. **Progressive Web App**: Works offline, installable on mobile
6. **Dual Theme**: Light (eco-tech) and dark modes

### System Statistics

- **8 Animal Types**: Dog, Cat, Cow, Horse, Sheep, Goat, Pig, Rabbit
- **35+ Health Indicators**: Symptoms, vital signs, duration
- **10+ Diseases**: Pre-loaded disease knowledge base
- **3 ML Models**: RandomForest, XGBoost, LightGBM per animal
- **2-Stage Prediction**: Syndrome → Disease

---

## Database Architecture

### Database: **Supabase (PostgreSQL)**

#### Core Tables (9 tables total)

### 1. **users** - User Accounts

**Purpose**: Stores both farmer and veterinarian accounts

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    farm_name VARCHAR(255),
    location VARCHAR(255),
    phone VARCHAR(50),
    user_type VARCHAR(20) DEFAULT 'farmer',
    -- Vet-specific fields
    clinic_name VARCHAR(255),
    license_number VARCHAR(100),
    specialization VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Key Points**:
- Single table for both user types (polymorphic)
- `user_type`: 'farmer' or 'veterinarian'
- Passwords hashed with bcrypt (cost factor 12)
- Vet-specific fields: clinic_name, license_number, specialization

**Sample Data**:
```sql
-- Vet account
Email: dr.sharma@pashucare.com
Password: VetPass123!
user_type: veterinarian

-- Farmer account  
Email: farmer.patil@pashucare.com
Password: FarmerPass123!
user_type: farmer
```

---

### 2. **animals** - Livestock Records

**Purpose**: Stores individual animal information

```sql
CREATE TABLE animals (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    animal_id VARCHAR(50) UNIQUE,
    animal_type VARCHAR(50) NOT NULL,
    breed VARCHAR(100),
    name VARCHAR(255),
    age FLOAT,
    gender VARCHAR(20),
    weight FLOAT,
    health_status VARCHAR(50) DEFAULT 'healthy',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Indexes**:
- `idx_animals_user_id` - Fast lookup by owner
- `idx_animals_type` - Filter by animal type

**Relationships**:
- `user_id` → `users.id` (Owner)
- CASCADE delete: When user deleted, animals deleted

**Sample Values**:
- animal_type: 'Dog', 'Cat', 'Cow', 'Horse', 'Sheep', 'Goat', 'Pig', 'Rabbit'
- health_status: 'healthy', 'sick', 'recovering', 'critical'
- breed: Varies by animal_type (e.g., 'Labrador' for Dog, 'Holstein' for Cow)

---

### 3. **farm_lands** - Farm Properties

**Purpose**: Tracks farm land parcels

```sql
CREATE TABLE farm_lands (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    land_name VARCHAR(255) NOT NULL,
    size_acres FLOAT,
    location VARCHAR(255),
    soil_type VARCHAR(100),
    crops_grown TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Features**:
- Track multiple land parcels per farmer
- Analytics dashboard per land
- Crop management

---

### 4. **predictions** - AI Prediction History

**Purpose**: Stores all AI disease prediction results

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    animal_id INTEGER REFERENCES animals(id) ON DELETE SET NULL,
    prediction_data JSONB,
    result JSONB,
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Key Features**:
- **JSONB columns**: Stores complex prediction data
- SET NULL on delete: Preserves predictions even if user/animal deleted
- confidence: 0.0 to 1.0 (percentage)

**prediction_data structure**:
```json
{
  "animal_type": "Dog",
  "breed": "Labrador",
  "age": 5,
  "symptoms": ["Coughing", "Fever", "Lethargy"],
  "vital_signs": {
    "temperature": 39.8,
    "heart_rate": 110
  }
}
```

**result structure**:
```json
{
  "predicted_disease": "Pneumonia",
  "confidence": 0.85,
  "syndrome": "Respiratory",
  "top_3_predictions": [
    {"disease": "Pneumonia", "probability": 0.85},
    {"disease": "Bronchitis", "probability": 0.10},
    {"disease": "Kennel Cough", "probability": 0.05}
  ],
  "vital_signs_analysis": {
    "temperature_status": "High",
    "heart_rate_status": "Normal"
  }
}
```

**Indexes**:
- `idx_predictions_user_id` - User's prediction history
- `idx_predictions_animal_id` - Animal's medical history

---

### 5. **diseases** - Disease Reference Database

**Purpose**: Knowledge base of animal diseases

```sql
CREATE TABLE diseases (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    symptoms TEXT,
    recommended_treatment TEXT,
    prevention_measures TEXT,
    severity VARCHAR(50) DEFAULT 'medium',
    animal_types TEXT,
    is_contagious BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Pre-loaded Diseases**:
1. **Foot and Mouth Disease** (Critical, Contagious)
2. **Mastitis** (High severity)
3. **Rabies** (Critical, Contagious)
4. **Canine Parvovirus** (Critical, Contagious)
5. **Feline Leukemia** (High severity)
6. **Equine Influenza** (Medium severity)
7. **Pneumonia** (High severity)
8. **Ringworm** (Low severity, Contagious)
9. **Bloat** (Critical)
10. **Coccidiosis** (Medium severity)

**Severity Levels**:
- `critical`: Immediate vet attention required
- `high`: Urgent treatment needed
- `medium`: Monitor closely
- `low`: Manageable at home

**Indexes**:
- `idx_diseases_name` - Fast disease lookup
- `idx_diseases_severity` - Filter by severity

---

### 6. **vaccinations** - Vaccination Records

**Purpose**: Tracks all vaccinations administered

```sql
CREATE TABLE vaccinations (
    id SERIAL PRIMARY KEY,
    animal_id INTEGER NOT NULL REFERENCES animals(id) ON DELETE CASCADE,
    vet_id INTEGER REFERENCES users(id) ON DELETE RESTRICT,
    vaccine_name VARCHAR(255) NOT NULL,
    dose VARCHAR(100),
    batch_number VARCHAR(100),
    vaccination_date DATE NOT NULL,
    next_due_date DATE,
    administered_by VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Key Features**:
- Links to veterinarian who administered (`vet_id`)
- Batch number tracking for recalls
- Next due date for reminders
- RESTRICT delete: Can't delete vet if they have vaccination records

**Common Vaccines**:
- Dogs: Rabies, Parvovirus, Distemper, Hepatitis
- Cats: Rabies, Feline Leukemia, Calicivirus
- Cattle: FMD, Brucellosis, Anthrax
- Horses: Tetanus, Influenza, EHV

**Indexes**:
- `idx_vaccinations_animal_id` - Animal's vaccination history
- `idx_vaccinations_vet_id` - Vet's administered vaccines
- `idx_vaccinations_date` - Upcoming due dates

---

### 7. **animal_diseases** - Diagnosis Records

**Purpose**: Links animals to diagnosed diseases

```sql
CREATE TABLE animal_diseases (
    id SERIAL PRIMARY KEY,
    animal_id INTEGER NOT NULL REFERENCES animals(id) ON DELETE CASCADE,
    disease_id INTEGER NOT NULL REFERENCES diseases(id) ON DELETE RESTRICT,
    diagnosed_by_vet_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    date_diagnosed DATE NOT NULL,
    severity VARCHAR(50) DEFAULT 'medium',
    status VARCHAR(50) DEFAULT 'active',
    symptoms_observed TEXT,
    treatment_given TEXT,
    notes TEXT,
    follow_up_date DATE,
    recovery_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Status Values**:
- `active`: Currently sick
- `recovering`: Treatment in progress
- `recovered`: Fully recovered
- `chronic`: Long-term condition

**Workflow**:
1. Vet diagnoses disease → Creates record with status='active'
2. Treatment begins → Updates treatment_given field
3. Follow-up scheduled → Sets follow_up_date
4. Animal recovers → Updates status='recovered', sets recovery_date

**Indexes**:
- `idx_animal_diseases_animal_id` - Medical history
- `idx_animal_diseases_vet_id` - Vet's diagnosis records
- `idx_animal_diseases_date` - Chronological order
- `idx_animal_diseases_status` - Filter active cases

---

### 8. **vet_appointments** - Appointment Scheduling

**Purpose**: Schedule appointments between vets and farmers

```sql
CREATE TABLE vet_appointments (
    id SERIAL PRIMARY KEY,
    animal_id INTEGER NOT NULL REFERENCES animals(id) ON DELETE CASCADE,
    vet_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    farmer_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    appointment_date TIMESTAMP WITH TIME ZONE NOT NULL,
    reason VARCHAR(255),
    status VARCHAR(50) DEFAULT 'scheduled',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Status Values**:
- `scheduled`: Appointment booked
- `confirmed`: Both parties confirmed
- `completed`: Visit finished
- `cancelled`: Appointment cancelled
- `no_show`: Patient didn't show up

**Indexes**:
- `idx_appointments_vet_id` - Vet's schedule
- `idx_appointments_date` - Calendar view
- `idx_appointments_status` - Filter upcoming

---

### 9. **veterinarians** - Public Vet Directory

**Purpose**: Directory of available veterinarians

```sql
CREATE TABLE veterinarians (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    specialization VARCHAR(255),
    location VARCHAR(255),
    phone VARCHAR(50),
    email VARCHAR(255),
    experience_years INTEGER,
    rating FLOAT DEFAULT 0,
    is_available BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Note**: This is separate from `users` table - public directory vs actual user accounts

---

### 10. **subsidies** - Government Schemes

**Purpose**: Government agricultural schemes/subsidies

```sql
CREATE TABLE subsidies (
    id SERIAL PRIMARY KEY,
    scheme_name VARCHAR(255) NOT NULL,
    scheme_type VARCHAR(100),
    state VARCHAR(100),
    description TEXT,
    eligibility TEXT,
    subsidy_amount VARCHAR(100),
    application_deadline DATE,
    contact_info VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

---

### Database Relationships Diagram

```
users (1) ───→ (many) animals
users (1) ───→ (many) farm_lands
users (1) ───→ (many) predictions

animals (1) ───→ (many) predictions
animals (1) ───→ (many) vaccinations
animals (1) ───→ (many) animal_diseases
animals (1) ───→ (many) vet_appointments

users [vet] (1) ───→ (many) vaccinations
users [vet] (1) ───→ (many) animal_diseases
users [vet] (1) ───→ (many) vet_appointments

diseases (1) ───→ (many) animal_diseases
```

---

### Row Level Security (RLS)

**All tables have RLS enabled** for security:

1. **users**: Users can view/update own profile
2. **animals**: Farmers see own animals, vets see all
3. **farm_lands**: Users manage own lands only
4. **predictions**: Users see own predictions
5. **vaccinations**: Authenticated users can CRUD
6. **animal_diseases**: Authenticated users can CRUD
7. **diseases**: Public read-only access
8. **vet_appointments**: Authenticated users manage own

---

## Application Architecture

### Tech Stack

**Backend**:
- **Flask 3.1.2**: Web framework
- **Python 3.8+**: Programming language
- **Supabase 2.25.0**: Database (PostgreSQL)
- **Flask-Login 0.6.3**: Session management
- **Flask-Bcrypt 1.0.1**: Password hashing
- **python-dotenv 1.2.1**: Environment config

**ML Stack**:
- **scikit-learn 1.7.2**: ML framework
- **XGBoost 3.1.2**: Gradient boosting
- **LightGBM 4.6.0**: Light gradient boosting
- **pandas 2.3.3**: Data processing
- **numpy 2.3.5**: Numerical computing
- **imbalanced-learn 0.14.0**: SMOTE & undersampling
- **joblib 1.5.2**: Model serialization

**Frontend**:
- **Bootstrap 5.3.0**: UI framework
- **Vanilla JavaScript ES6+**: Interactivity
- **Custom CSS**: Eco-tech theme with dark mode
- **Font Awesome 6.4.0**: Icons

**Additional**:
- **pyttsx3 2.99**: Text-to-speech
- **SpeechRecognition 3.14.4**: Voice input

---

### Application Structure

```
app.py (3600+ lines)
├── Configuration & Setup
│   ├── Flask app initialization
│   ├── Supabase connection
│   ├── Login manager setup
│   └── Environment variables
│
├── Models & Classes
│   ├── User (Flask-Login UserMixin)
│   ├── DB (Supabase helper methods)
│   └── AnimalSpecificDiseasePredictor (ML model)
│
├── Utilities
│   ├── Language system (LANGUAGES dict)
│   ├── Template filters (from_json, format_date)
│   └── Voice quiz questions
│
├── Routes (50+ routes)
│   ├── Public routes
│   ├── Authentication routes
│   ├── Farmer dashboard routes
│   ├── Veterinarian routes
│   ├── API endpoints
│   └── PWA routes
│
└── Main execution
    ├── Load/train ML models
    └── Start Flask server
```

---

## Complete Route Documentation

### Public Routes (No Authentication)

#### 1. **Homepage** - `/`
```python
@app.route('/')
def index()
```
- **Purpose**: Main landing page with AI prediction form
- **Template**: `index.html`
- **Features**: 
  - Disease prediction form
  - Quick actions
  - Farmer/vet statistics
  - Language toggle

#### 2. **Health Assessment** - `/health_assessment`
```python
@app.route('/health_assessment')
def health_assessment()
```
- **Purpose**: Standalone health assessment page
- **Template**: `health_assessment.html`
- **Data**: Animal types, breeds, symptoms

#### 3. **Knowledge Base** - `/knowledge_base`
```python
@app.route('/knowledge_base')
def knowledge_base()
```
- **Purpose**: Disease information library
- **Template**: `knowledge_base.html`
- **Data**: All diseases from database

#### 4. **Veterinarians Directory** - `/veterinarians`
```python
@app.route('/veterinarians')
def veterinarians()
```
- **Purpose**: Find local veterinarians
- **Template**: `veterinarians.html`
- **Features**: 
  - Search by location
  - Contact info (phone, email, WhatsApp)
  - Ratings and specializations

#### 5. **Government Subsidies** - `/subsidies`
```python
@app.route('/subsidies')
def subsidies()
```
- **Purpose**: Browse government schemes
- **Template**: `subsidies.html`
- **Data**: Active subsidies filtered by state

#### 6. **Voice Quiz** - `/voice_quiz`
```python
@app.route('/voice_quiz')
def voice_quiz()
```
- **Purpose**: Voice-based health assessment
- **Template**: `voice_quiz.html`
- **Features**: 15 question voice quiz

#### 7. **Offline Page** - `/offline.html`
```python
@app.route('/offline.html')
def offline()
```
- **Purpose**: PWA offline fallback
- **Template**: `offline.html`

---

### Authentication Routes

#### 8. **Register** - `/register` [GET, POST]
```python
@app.route('/register', methods=['GET', 'POST'])
def register()
```
- **GET**: Display registration form
- **POST**: Create new user account
  - Validates email uniqueness
  - Hashes password with bcrypt
  - Creates user in Supabase
  - Auto-login after registration
- **Template**: `auth/register.html`
- **Redirect**: Dashboard on success

#### 9. **Login** - `/login` [GET, POST]
```python
@app.route('/login', methods=['GET', 'POST'])
def login()
```
- **GET**: Display login form
- **POST**: Authenticate user
  - Validates email/password
  - Checks bcrypt hash
  - Creates Flask-Login session
  - Redirects based on user_type
- **Template**: `auth/login.html`
- **Redirect**: 
  - Farmer → `/farmer/dashboard`
  - Vet → `/vet/dashboard`

#### 10. **Logout** - `/logout`
```python
@app.route('/logout')
@login_required
def logout()
```
- **Purpose**: End user session
- **Method**: Flask-Login logout_user()
- **Redirect**: Homepage

---

### Farmer Dashboard Routes (Authentication Required)

#### 11. **Farmer Dashboard** - `/farmer/dashboard`
```python
@app.route('/farmer/dashboard')
@login_required
def farmer_dashboard()
```
- **Purpose**: Farmer overview
- **Template**: `dashboard/main.html`
- **Data**:
  - Total animals count
  - Total lands count
  - Sick animals count
  - Recent predictions (last 5)
  - Animal type breakdown
- **Stats Calculation**:
  ```python
  total_animals = len(DB.get_user_animals(current_user.id))
  sick_animals = [a for a in animals if a['health_status'] == 'sick']
  ```

#### 12. **Animals List** - `/animals`
```python
@app.route('/animals')
@login_required
def animals()
```
- **Purpose**: View all owned animals
- **Template**: `dashboard/animals.html`
- **Features**: Animal cards with quick actions

#### 13. **Add Animal** - `/add_animal` [GET, POST]
```python
@app.route('/add_animal', methods=['GET', 'POST'])
@login_required
def add_animal()
```
- **GET**: Display form
- **POST**: Create animal record
  - Validates required fields
  - Generates unique animal_id
  - Stores in Supabase
- **Template**: `dashboard/add_animal.html`
- **Fields**: animal_type, breed, name, age, gender, weight, health_status

#### 14. **Animal Detail** - `/animal/<int:animal_id>`
```python
@app.route('/animal/<int:animal_id>')
@login_required
def animal_detail(animal_id)
```
- **Purpose**: View animal profile
- **Template**: `dashboard/animal_detail.html`
- **Data**:
  - Animal info
  - Vaccination history (last 10)
  - Prediction history (last 10)
  - Health timeline

#### 15. **Edit Animal** - `/animal/<int:animal_id>/edit` [GET, POST]
```python
@app.route('/animal/<int:animal_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_animal(animal_id)
```
- **GET**: Display pre-filled form
- **POST**: Update animal record
- **Template**: `dashboard/edit_animal.html`
- **Validation**: Ownership check

#### 16. **Predict for Animal** - `/animal/<int:animal_id>/predict` [GET, POST]
```python
@app.route('/animal/<int:animal_id>/predict', methods=['GET', 'POST'])
@login_required
def predict_for_animal(animal_id)
```
- **Purpose**: Quick health check for specific animal
- **GET**: Pre-filled form with animal data
- **POST**: Run AI prediction
- **Template**: `dashboard/animal_predict.html`
- **Features**: Auto-fills breed, age, gender, weight

#### 17. **Animal Vaccinations** - `/animal/<int:animal_id>/vaccinations`
```python
@app.route('/animal/<int:animal_id>/vaccinations')
@login_required
def animal_vaccinations(animal_id)
```
- **Purpose**: View vaccination records
- **Template**: `dashboard/vaccinations.html`
- **Data**: All vaccinations for animal, sorted by date DESC

#### 18. **Add Vaccination** - `/animal/<int:animal_id>/add_vaccination` [GET, POST]
```python
@app.route('/animal/<int:animal_id>/add_vaccination', methods=['GET', 'POST'])
@login_required
def add_vaccination(animal_id)
```
- **Purpose**: Record new vaccination
- **Template**: `dashboard/add_vaccination.html`
- **Fields**: vaccine_name, dose, vaccination_date, next_due_date, administered_by, notes

#### 19. **Farm Lands List** - `/lands`
```python
@app.route('/lands')
@login_required
def lands()
```
- **Purpose**: View owned land parcels
- **Template**: `dashboard/lands.html`

#### 20. **Add Land** - `/add_land` [GET, POST]
```python
@app.route('/add_land', methods=['GET', 'POST'])
@login_required
def add_land()
```
- **Purpose**: Register new farm land
- **Template**: `dashboard/add_land.html`
- **Fields**: land_name, size_acres, location, soil_type, crops_grown

#### 21. **Edit Land** - `/land/<int:land_id>/edit` [GET, POST]
```python
@app.route('/land/<int:land_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_land(land_id)
```
- **Purpose**: Update land details
- **Template**: `dashboard/edit_land.html`

#### 22. **Land Analytics** - `/land/<int:land_id>/analytics`
```python
@app.route('/land/<int:land_id>/analytics')
@login_required
def land_analytics(land_id)
```
- **Purpose**: View land performance metrics
- **Template**: `dashboard/land_analytics.html`
- **Features**: Productivity charts, resource usage

#### 23. **Profile** - `/profile`
```python
@app.route('/profile')
@login_required
def profile()
```
- **Purpose**: View/edit user profile
- **Template**: `dashboard/profile.html`

#### 24. **Update Profile** - `/update_profile` [POST]
```python
@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile()
```
- **Purpose**: Save profile changes
- **Fields**: name, email, phone, farm_name, location, password (optional)
- **Redirect**: `/profile`

---

### Veterinarian Routes (Authentication Required)

#### 25. **Vet Dashboard** - `/vet/dashboard`
```python
@app.route('/vet/dashboard')
@login_required
def vet_dashboard()
```
- **Purpose**: Vet overview with stats
- **Template**: `vet/dashboard.html`
- **Stats**:
  - Total animals treated (unique count)
  - Total vaccinations given
  - Total diagnoses made
  - Recent animals (last 10)
  - Recent vaccinations (last 10)
- **Features**: Search bar for animal lookup

#### 26. **Animal Search** - `/vet/animal/search` [GET, POST]
```python
@app.route('/vet/animal/search', methods=['GET', 'POST'])
@login_required
def vet_animal_search()
```
- **GET**: Display search form
- **POST**: Execute search
  - Search by ID, name, or animal type
  - Shows owner information
- **Template**: `vet/search.html`
- **Query**: 
  ```python
  DB.search_animals(query)  # ilike search on name/animal_type
  ```

#### 27. **Vet Animal Detail** - `/vet/animal/<int:animal_id>`
```python
@app.route('/vet/animal/<int:animal_id>')
@login_required
def vet_animal_detail(animal_id)
```
- **Purpose**: View complete animal profile
- **Template**: `vet/animal_detail.html`
- **Data**:
  - Animal info + owner info
  - Vaccination history
  - Diagnosis history
  - Forms to add records
- **Features**: Add vaccination, add diagnosis

#### 28. **Add Vaccination (Vet)** - `/vet/vaccinate` [POST]
```python
@app.route('/vet/vaccinate', methods=['POST'])
@login_required
def vet_add_vaccination()
```
- **Purpose**: Record vaccination administered by vet
- **Fields**: animal_id, vaccine_name, dose, batch_number, vaccination_date, next_due_date, notes
- **Logic**: 
  ```python
  data['vet_id'] = current_user.id
  DB.add_vaccination_by_vet(vet_id, animal_id, data)
  ```
- **Redirect**: Back to animal detail page

#### 29. **Edit Vaccination** - `/vet/vaccinate/<int:vac_id>/edit` [POST]
```python
@app.route('/vet/vaccinate/<int:vac_id>/edit', methods=['POST'])
@login_required
def vet_edit_vaccination(vac_id)
```
- **Purpose**: Update vaccination record
- **Authorization**: Only vet who created it
- **Redirect**: Referrer

#### 30. **Delete Vaccination** - `/vet/vaccinate/<int:vac_id>/delete` [POST]
```python
@app.route('/vet/vaccinate/<int:vac_id>/delete', methods=['POST'])
@login_required
def vet_delete_vaccination(vac_id)
```
- **Purpose**: Remove vaccination record
- **Authorization**: Only vet who created it
- **Redirect**: Referrer

#### 31. **Add Diagnosis** - `/vet/diagnose` [POST]
```python
@app.route('/vet/diagnose', methods=['POST'])
@login_required
def vet_add_diagnosis()
```
- **Purpose**: Create diagnosis record
- **Fields**: animal_id, disease_id, date_diagnosed, severity, status, symptoms_observed, treatment_given, follow_up_date, notes
- **Logic**:
  ```python
  data['diagnosed_by_vet_id'] = current_user.id
  DB.add_diagnosis(vet_id, animal_id, disease_id, data)
  ```

#### 32. **Update Diagnosis** - `/vet/diagnose/<int:diagnosis_id>/update` [POST]
```python
@app.route('/vet/diagnose/<int:diagnosis_id>/update', methods=['POST'])
@login_required
def vet_update_diagnosis(diagnosis_id)
```
- **Purpose**: Update diagnosis status/treatment
- **Fields**: status, treatment_given, recovery_date, notes
- **Use Case**: Mark as 'recovered' when animal heals

#### 33. **Vaccinations List** - `/vet/vaccinations`
```python
@app.route('/vet/vaccinations')
@login_required
def vet_vaccinations_list()
```
- **Purpose**: View all vaccinations administered by this vet
- **Template**: `vet/vaccinations_list.html`
- **Query**: `DB.get_vaccinations_by_vet(current_user.id)`

---

### AI Prediction Routes

#### 34. **Predict Disease** - `/predict` [POST]
```python
@app.route('/predict', methods=['POST'])
def predict()
```
- **Purpose**: Main AI prediction endpoint
- **Method**: POST only
- **Authentication**: Optional (works for guests)
- **Input Fields** (from form):
  ```python
  animal_type, breed, age, gender, weight,
  symptom1, symptom2, symptom3, symptom4, duration,
  appetite_loss, vomiting, diarrhea, coughing,
  labored_breathing, lameness, skin_lesions,
  nasal_discharge, eye_discharge,
  body_temperature, heart_rate
  ```
- **Process**:
  1. Validate inputs
  2. Call `predictor.predict_disease(...)`
  3. Get top 3 predictions with confidence
  4. Get syndrome classification
  5. Get vital signs analysis
  6. Save to database (if logged in)
- **Response**: Renders `result.html` with prediction
- **Error Handling**: Returns error message if model fails

**Prediction Flow**:
```
User Input → Feature Engineering → Syndrome Prediction → Disease Prediction → Results
```

---

### API Endpoints (JSON Responses)

#### 35. **Get Breeds** - `/get_breeds/<animal_type>`
```python
@app.route('/get_breeds/<animal_type>')
def get_breeds(animal_type)
```
- **Purpose**: Dynamic breed dropdown
- **Response**:
  ```json
  [
    ["Labrador", "Labrador"],
    ["German Shepherd", "जर्मन शेफर्ड"]
  ]
  ```
- **Language**: Returns translated breed names based on session language

#### 36. **Get Symptom Translations** - `/get_symptoms_translated`
```python
@app.route('/get_symptoms_translated')
def get_symptoms_translated()
```
- **Purpose**: Get symptom translations for current language
- **Response**: JSON dict of symptom translations

#### 37. **Model Status** - `/model_status`
```python
@app.route('/model_status')
def model_status()
```
- **Purpose**: Check ML model availability and metrics
- **Response**:
  ```json
  {
    "loaded": true,
    "available_animals": ["Dog", "Cat", "Cow", ...],
    "model_metrics": {
      "overall_accuracy": 0.85,
      "total_samples": 1250,
      "animal_metrics": {
        "Dog": {
          "accuracy": 0.847,
          "precision": 0.831,
          "samples": 450
        }
      }
    }
  }
  ```

#### 38. **Dashboard Data API** - `/api/dashboard_data`
```python
@app.route('/api/dashboard_data')
@login_required
def dashboard_data()
```
- **Purpose**: Get dashboard stats as JSON
- **Response**:
  ```json
  {
    "total_animals": 15,
    "total_lands": 3,
    "sick_animals": 2,
    "predictions_count": 45,
    "animal_types": {
      "Cow": 8,
      "Goat": 4,
      "Dog": 3
    },
    "recent_predictions": [...]
  }
  ```

---

### Voice Quiz Routes

#### 39. **Voice Quiz Start** - `/voice_quiz_start` [POST]
```python
@app.route('/voice_quiz_start', methods=['POST'])
def voice_quiz_start()
```
- **Purpose**: Initialize voice quiz session
- **Response**: Session ID and first question

#### 40. **Voice Quiz Listen** - `/voice_quiz_listen` [POST]
```python
@app.route('/voice_quiz_listen', methods=['POST'])
def voice_quiz_listen()
```
- **Purpose**: Process voice input
- **Uses**: SpeechRecognition library
- **Response**: Transcribed text

#### 41. **Voice Quiz Submit** - `/voice_quiz_submit` [POST]
```python
@app.route('/voice_quiz_submit', methods=['POST'])
def voice_quiz_submit()
```
- **Purpose**: Submit all quiz answers for prediction
- **Logic**: Converts quiz answers to prediction format
- **Response**: Prediction result

#### 42. **Voice Quiz Speak** - `/voice_quiz_speak` [POST]
```python
@app.route('/voice_quiz_speak', methods=['POST'])
def voice_quiz_speak()
```
- **Purpose**: Text-to-speech for question
- **Uses**: pyttsx3 library
- **Response**: Success/failure

#### 43. **Voice Quiz Stop** - `/voice_quiz_stop` [POST]
```python
@app.route('/voice_quiz_stop', methods=['POST'])
def voice_quiz_stop()
```
- **Purpose**: Stop TTS playback

#### 44. **Voice Quiz Result** - `/voice_quiz_result`
```python
@app.route('/voice_quiz_result')
def voice_quiz_result()
```
- **Purpose**: Display quiz prediction result
- **Template**: Same as regular prediction result

---

### Utility Routes

#### 45. **Set Language** - `/set_language/<language>`
```python
@app.route('/set_language/<language>')
def set_language(language)
```
- **Purpose**: Switch between English/Marathi
- **Values**: 'en' or 'mr'
- **Storage**: Session cookie
- **Redirect**: Referrer (stays on same page)

#### 46. **Service Worker** - `/sw.js`
```python
@app.route('/sw.js')
def service_worker()
```
- **Purpose**: Serve PWA service worker
- **File**: `static/sw.js`
- **Features**: Offline caching, background sync

#### 47. **Upload Image** - `/upload_image` [POST]
```python
@app.route('/upload_image', methods=['POST'])
def upload_image()
```
- **Purpose**: Upload animal images (future feature)
- **Status**: Placeholder for image-based diagnosis

---

## Frontend Architecture

### Theme System

**Dual Theme**: Light (Eco-Tech) + Dark (AgriTech)

#### CSS Variables (Light Mode)

```css
:root {
  /* Primary Colors */
  --primary: #00695C;       /* Teal Green */
  --secondary: #A7FFEB;     /* Light Mint */
  --accent: #FBC02D;        /* Harvest Yellow */
  
  /* Text Colors */
  --text-primary: #263238;  /* Charcoal Grey */
  --background: #FAFAFA;    /* Soft White */
  
  /* Gradients */
  --gradient-primary: linear-gradient(135deg, #00695C 0%, #26A69A 100%);
}
```

#### Dark Mode Variables

```css
[data-theme="dark"] {
  --primary: #26A69A;       /* Mint Green */
  --background: #1A1A1A;    /* Deep Charcoal */
  --text-primary: #E0E0E0;  /* Light Grey */
  --card-bg: #2A2A2A;       /* Dark Card */
}
```

**Theme Toggle**:
- Button in navbar
- Persists in localStorage
- Smooth transition: `transition: all 0.3s ease`

---

### JavaScript Architecture

**File**: `static/scripts.js`

**Class**: `PashuCareEnhanced`

```javascript
class PashuCareEnhanced {
  constructor() {
    this.init();
    this.setupAnimations();
    this.setupInteractions();
    this.setupFormValidation();
    this.setupTheme();
  }
  
  // Methods
  setupAnimalBreedHandler()    // Dynamic breed loading
  setupScrollAnimations()      // Intersection Observer
  setupParticleBackground()    // Animated background
  setupInteractions()          // Click/hover effects
  createRippleEffect()         // Material Design ripple
  setupFormValidation()        // Real-time validation
  validateField()              // Single field validation
  validateForm()               // Full form validation
  showLoadingOverlay()         // Loading indicator
  showNotification()           // Toast notifications
  animateElement()             // CSS animation triggers
  toggleSymptomCard()          // Symptom selection
}
```

**Key Features**:

1. **Dynamic Breed Loading**:
   ```javascript
   animalSelect.addEventListener('change', async (e) => {
     const response = await fetch(`/get_breeds/${animalType}`);
     const breeds = await response.json();
     this.populateBreeds(breedSelect, breeds);
   });
   ```

2. **Real-time Validation**:
   ```javascript
   field.addEventListener('blur', () => {
     this.validateField(field);
   });
   ```

3. **Scroll Animations**:
   ```javascript
   const observer = new IntersectionObserver((entries) => {
     entries.forEach(entry => {
       if (entry.isIntersecting) {
         entry.target.classList.add('animate-in');
       }
     });
   });
   ```

---

### Progressive Web App (PWA)

**Manifest**: `static/manifest.json`

```json
{
  "name": "PashuCare - AI Livestock Health",
  "short_name": "PashuCare",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#00695C",
  "theme_color": "#00695C",
  "icons": [
    {
      "src": "/static/favicon.ico",
      "sizes": "192x192",
      "type": "image/png"
    }
  ]
}
```

**Service Worker**: `static/sw.js`

**Features**:
- Cache static assets (CSS, JS, images)
- Offline fallback page
- Background sync for predictions
- Install prompt

---

## User Roles & Workflows

### Role 1: Guest (No Authentication)

**Capabilities**:
- ✅ Use AI disease prediction
- ✅ Browse knowledge base
- ✅ View veterinarian directory
- ✅ Check government subsidies
- ✅ Use voice quiz
- ❌ Save predictions
- ❌ Manage animals
- ❌ Access dashboard

**Typical Workflow**:
1. Visit homepage
2. Fill prediction form
3. Get instant diagnosis
4. View recommendations
5. Find local vet
6. (Optional) Register account

---

### Role 2: Farmer

**Capabilities**:
- ✅ All guest features
- ✅ Create account
- ✅ Add/manage animals
- ✅ Add/manage farm lands
- ✅ Save prediction history
- ✅ Track vaccinations
- ✅ Schedule vet appointments
- ✅ View analytics
- ❌ Access other farmers' data
- ❌ Add diagnoses

**Typical Workflow**:

**1. Registration & Setup**:
```
Register → Create Account → Add Animals → Add Farm Lands
```

**2. Daily Monitoring**:
```
Login → Dashboard → View Animal Health Status → Run Health Check (if needed)
```

**3. Health Issue Detected**:
```
Notice Symptoms → Run AI Prediction → Review Results → 
  ├─ Minor: Follow Recommendations
  └─ Serious: Find Vet → Schedule Appointment
```

**4. Veterinary Visit**:
```
Vet Examines Animal → Vet Adds Vaccination/Diagnosis → 
Farmer Views Updated Records → Follow Treatment Plan
```

**5. Record Keeping**:
```
Add Vaccination Records → Track Recovery → 
Update Health Status → View History
```

---

### Role 3: Veterinarian

**Capabilities**:
- ✅ All guest features
- ✅ Vet-specific dashboard
- ✅ Search any animal by ID/name
- ✅ View animal medical history
- ✅ Add vaccination records
- ✅ Create diagnoses
- ✅ Update treatment plans
- ✅ Track statistics
- ✅ View all patients
- ❌ Modify other vets' records
- ❌ Delete animals

**Typical Workflow**:

**1. Daily Practice**:
```
Login → Vet Dashboard → View Today's Appointments → 
Review Statistics (patients treated, vaccines given)
```

**2. Animal Consultation**:
```
Farmer Brings Animal → Search Animal by ID → 
View Medical History → Examine Animal → 
  ├─ Vaccinate → Add Vaccination Record
  └─ Diagnose → Add Diagnosis + Treatment Plan
```

**3. Follow-up Visit**:
```
Search Animal → View Previous Diagnosis → 
Check Progress → Update Diagnosis Status → 
  ├─ Still Sick: Adjust Treatment
  ├─ Recovering: Schedule Follow-up
  └─ Recovered: Mark as 'recovered'
```

**4. Record Management**:
```
View Vaccinations List → Edit/Delete Own Records → 
View Diagnosis History → Generate Reports
```

---

## Feature Deep Dive

### Feature 1: AI Disease Prediction

**Entry Points**:
1. Homepage form
2. Animal detail page (quick check)
3. Voice quiz

**Prediction Process**:

**Step 1: Data Collection**
```
User Inputs:
├─ Animal Info: type, breed, age, gender, weight
├─ Symptoms: Primary (4) + Observable (9)
├─ Duration: How many days
└─ Vital Signs: temperature, heart rate
```

**Step 2: Feature Engineering** (35+ features created)
```python
# Species-specific vital sign analysis
normal_ranges = {
  'Dog': {'temp': (38.0, 39.2), 'hr': (60, 160)},
  'Cat': {'temp': (38.1, 39.2), 'hr': (140, 220)},
  ...
}

# Syndrome scores
respiratory_score = coughing*3 + labored_breathing*4 + nasal_discharge*2
gi_score = vomiting*4 + diarrhea*3 + appetite_loss*2
systemic_score = temp_abnormal*3 + appetite_loss*2

# Condition flags
acute_condition = 1 if duration <= 3 else 0
chronic_condition = 1 if duration > 14 else 0
multi_system = 1 if affected_systems >= 2 else 0
```

**Step 3: Two-Stage Prediction**

**Stage 1: Syndrome Classification**
```
Input Features → RandomForest Classifier → Syndrome
```
Possible syndromes:
- Respiratory
- GI (Gastrointestinal)
- Multi-System
- Dermatological
- Neurological
- Systemic

**Stage 2: Disease Classification** (per syndrome)
```
Syndrome + Features → Ensemble (RF + XGB + LGB) → Disease
```

**Step 4: Ensemble Averaging**
```python
# Average probabilities from 3 models
rf_probs = rf_model.predict_proba(X)
xgb_probs = xgb_model.predict_proba(X)
lgb_probs = lgb_model.predict_proba(X)

avg_probs = (rf_probs + xgb_probs + lgb_probs) / 3

# Get top 3 predictions
top_3_indices = np.argsort(avg_probs)[::-1][:3]
top_3_diseases = [disease_encoder.inverse_transform([i])[0] 
                  for i in top_3_indices]
```

**Step 5: Confidence Adjustment**
```python
# Weighted average of syndrome and disease confidence
final_confidence = 0.7 * disease_conf + 0.3 * syndrome_conf

# Boost high-confidence predictions
if disease_conf > 0.75 and syndrome_conf > 0.6:
    final_confidence = max(final_confidence, 0.75)
```

**Step 6: Result Assembly**
```python
result = {
  'predicted_disease': 'Pneumonia',
  'confidence': 0.85,
  'syndrome': 'Respiratory',
  'syndrome_confidence': 0.92,
  'top_3_predictions': [
    {'disease': 'Pneumonia', 'probability': 0.85},
    {'disease': 'Bronchitis', 'probability': 0.10},
    {'disease': 'Kennel Cough', 'probability': 0.05}
  ],
  'vital_signs_analysis': {
    'temperature_status': 'High',
    'heart_rate_status': 'Normal'
  },
  'syndrome_analysis': {
    'respiratory_score': 9,
    'gi_score': 2,
    'systemic_score': 3,
    'multi_system': False
  },
  'condition_severity': 'Acute'
}
```

**Step 7: Recommendations**

Based on prediction, generate:
- Immediate actions
- Treatment options
- Prevention measures
- When to seek vet help

---

### Feature 2: Bilingual Support

**Languages**: English (en) + Marathi (mr)

**Implementation**:

**1. Language Dictionary** (in `app.py`):
```python
LANGUAGES = {
  'en': {
    'app_name': 'PashuCare',
    'home': 'Home',
    'dashboard': 'Dashboard',
    ...
  },
  'mr': {
    'app_name': 'पशुकेअर',
    'home': 'मुख्यपृष्ठ',
    'dashboard': 'डॅशबोर्ड',
    ...
  }
}
```

**2. Context Processor**:
```python
@app.context_processor
def inject_language():
    return dict(
        get_text=get_text, 
        current_language=get_language()
    )
```

**3. Template Usage**:
```html
<h1>{{ get_text('welcome') }}</h1>
<!-- English: Welcome to PashuCare -->
<!-- Marathi: पशुकेअर मध्ये आपले स्वागत -->
```

**4. Language Toggle**:
```html
<a href="/set_language/mr">मराठी</a>
<a href="/set_language/en">English</a>
```

**5. Dynamic Content**:
- Breed names translated
- Symptom names translated
- Disease descriptions (partial)
- UI labels fully translated

---

### Feature 3: Veterinary Portal

**Unique Features for Vets**:

**1. Search Any Animal**:
```python
# By ID
animals = DB.search_animals("12345")

# By name
animals = DB.search_animals("Bella")

# By type
animals = DB.search_animals("Cow")
```

**2. View Owner Information**:
```sql
SELECT animals.*, users.name, users.email, users.phone
FROM animals
JOIN users ON animals.user_id = users.id
WHERE animals.id = $animal_id
```

**3. Add Medical Records**:

**Vaccination**:
```python
{
  'animal_id': 123,
  'vet_id': current_user.id,
  'vaccine_name': 'Rabies',
  'dose': '1ml',
  'batch_number': 'RAB-2024-001',
  'vaccination_date': '2024-01-15',
  'next_due_date': '2025-01-15'
}
```

**Diagnosis**:
```python
{
  'animal_id': 123,
  'disease_id': 5,  # Rabies
  'diagnosed_by_vet_id': current_user.id,
  'date_diagnosed': '2024-01-10',
  'severity': 'critical',
  'status': 'active',
  'symptoms_observed': 'Aggression, excessive drooling',
  'treatment_given': 'Quarantine, supportive care',
  'follow_up_date': '2024-01-17'
}
```

**4. Statistics Dashboard**:
```python
stats = {
  'total_vaccinations': 150,
  'total_diagnoses': 45,
  'unique_animals': 87,
  'this_month_vaccinations': 23,
  'active_cases': 12
}
```

---

## Technology Stack Details

### Backend Deep Dive

**Flask 3.1.2**:
- **Routing**: Decorator-based (`@app.route`)
- **Templating**: Jinja2 (with filters, context processors)
- **Sessions**: Flask-Session (secure cookies)
- **Error Handling**: Custom error pages

**Supabase 2.25.0**:
- **Database**: PostgreSQL 15+
- **Features**: 
  - Row Level Security (RLS)
  - Real-time subscriptions
  - Auto-generated REST API
  - Built-in authentication (not used, using Flask-Login instead)
- **Client**:
  ```python
  from supabase import create_client
  supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
  
  # Query
  response = supabase.table('animals').select('*').execute()
  data = response.data
  ```

**Flask-Login 0.6.3**:
- **User Management**: UserMixin class
- **Session Management**: `login_user()`, `logout_user()`
- **Protection**: `@login_required` decorator
- **Current User**: `current_user` object
- **Remember Me**: Cookie-based persistence

**Flask-Bcrypt 1.0.1**:
- **Hashing**: bcrypt algorithm
- **Cost Factor**: 12 (2^12 iterations)
- **Usage**:
  ```python
  # Hash
  password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
  
  # Verify
  is_valid = bcrypt.check_password_hash(password_hash, password)
  ```

---

### ML Stack Deep Dive

**scikit-learn 1.7.2**:
- **Models**: RandomForestClassifier
- **Preprocessing**: StandardScaler, LabelEncoder
- **Metrics**: accuracy_score, precision_score, recall_score, f1_score
- **Calibration**: CalibratedClassifierCV

**XGBoost 3.1.2**:
- **Model**: XGBClassifier
- **Parameters**: n_estimators=150, max_depth=6
- **Advantages**: 
  - Handles missing values
  - Feature importance
  - Fast training
  - Regularization (L1, L2)

**LightGBM 4.6.0**:
- **Model**: LGBMClassifier
- **Parameters**: n_estimators=150, max_depth=8
- **Advantages**:
  - Very fast
  - Low memory
  - Leaf-wise growth
  - Categorical feature support

**imbalanced-learn 0.14.0**:
- **SMOTE**: Synthetic Minority Over-sampling
  ```python
  from imblearn.over_sampling import SMOTE
  smote = SMOTE(random_state=42, k_neighbors=3)
  X_resampled, y_resampled = smote.fit_resample(X, y)
  ```
- **RandomUnderSampler**: Reduce majority class
  ```python
  from imblearn.under_sampling import RandomUnderSampler
  rus = RandomUnderSampler(random_state=42)
  X_resampled, y_resampled = rus.fit_resample(X, y)
  ```

**joblib 1.5.2**:
- **Model Serialization**:
  ```python
  # Save
  joblib.dump(model, 'model.joblib')
  
  # Load
  model = joblib.load('model.joblib')
  ```
- **Advantages**: Optimized for numpy arrays, compression

---

### Frontend Deep Dive

**Bootstrap 5.3.0**:
- **Grid System**: 12-column responsive
- **Components**: Cards, buttons, forms, navbar, modals
- **Utilities**: Spacing, colors, typography
- **JavaScript**: Dropdowns, modals, tooltips

**Custom CSS**:
- **Variables**: 50+ CSS custom properties
- **Gradients**: 6 pre-defined gradients
- **Animations**: Fade, slide, bounce, pulse
- **Transitions**: Smooth 0.3s cubic-bezier

**JavaScript ES6+**:
- **Features Used**:
  - Classes
  - Async/await
  - Arrow functions
  - Template literals
  - Destructuring
  - Intersection Observer API
  - Fetch API
- **No jQuery**: Pure vanilla JS

---

## Summary

**PashuCare** is a production-ready, full-stack web application that combines:

1. **Modern Web Technologies**: Flask, PostgreSQL, Bootstrap
2. **Advanced ML**: Hierarchical ensemble models with 85%+ accuracy
3. **Real-world Utility**: Serves farmers and veterinarians
4. **Bilingual Support**: English + Marathi
5. **Professional Features**: User management, role-based access, PWA
6. **Scalable Architecture**: Modular design, cloud database, RESTful APIs

**Key Metrics**:
- **3,600+ lines** of Python code
- **50+ routes** (REST endpoints)
- **10 database tables** with RLS
- **8 animal types** supported
- **35+ features** per prediction
- **3 ML models** per animal
- **2-stage prediction** system
- **100% training accuracy** (with calibration)

**Use Cases**:
- ✅ Early disease detection
- ✅ Farm management
- ✅ Veterinary practice management
- ✅ Health record keeping
- ✅ Government scheme awareness
- ✅ Vet directory

This system is ready for deployment and real-world use in agricultural settings, particularly in India where bilingual support is crucial.
