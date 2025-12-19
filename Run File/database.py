import mysql.connector
from mysql.connector import Error, pooling
import os
from datetime import datetime, date
import json
import logging
from typing import Optional, Dict, List, Any, Tuple
from contextlib import contextmanager
import uuid
from config import DatabaseConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Main database manager for health prediction system"""
    
    def __init__(self):
        self.config = DatabaseConfig()
        self.connection_pool = None
        self.init_database()
        self.init_connection_pool()
    
    def init_connection_pool(self):
        """Initialize connection pool"""
        try:
            self.connection_pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name=self.config.pool_name,
                pool_size=self.config.pool_size,
                host=self.config.host,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database
            )
            logger.info("✅ Connection pool initialized successfully")
        except Error as e:
            logger.error(f"❌ Failed to initialize connection pool: {e}")
            self.connection_pool = None
    
    def get_connection(self):
        """Get a connection from the pool"""
        try:
            if self.connection_pool:
                return self.connection_pool.get_connection()
            else:
                # Fallback to direct connection
                return mysql.connector.connect(**self.config.get_connection_params())
        except Error as e:
            logger.error(f"❌ Error getting database connection: {e}")
            return None
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        connection = None
        try:
            connection = self.get_connection()
            yield connection
        except Error as e:
            logger.error(f"❌ Database error: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    def init_database(self):
        """Initialize database and create tables if they don't exist"""
        try:
            # First, connect without database to create it if needed
            create_conn = mysql.connector.connect(
                host=self.config.host,
                user=self.config.user,
                password=self.config.password,
                port=self.config.port
            )
            create_cursor = create_conn.cursor()
            
            # Create database if not exists
            create_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.config.database}")
            create_cursor.execute(f"USE {self.config.database}")
            
            # Create tables
            self._create_tables(create_cursor)
            
            create_conn.commit()
            create_cursor.close()
            create_conn.close()
            
            logger.info("✅ Database tables initialized successfully")
            
        except Error as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    def _create_tables(self, cursor):
        """Create all necessary tables"""
        
        # Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id VARCHAR(50) PRIMARY KEY,
                first_name VARCHAR(100),
                last_name VARCHAR(100),
                age INT,
                gender ENUM('Male', 'Female', 'Other'),
                contact_number VARCHAR(20),
                email VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_visit TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                total_visits INT DEFAULT 0,
                address TEXT,
                emergency_contact VARCHAR(20),
                blood_group VARCHAR(5),
                allergies TEXT,
                medical_history TEXT,
                INDEX idx_last_visit (last_visit),
                INDEX idx_total_visits (total_visits)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Health records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_records (
                record_id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id VARCHAR(50),
                visit_number INT,
                
                -- Vital Signs
                temperature DECIMAL(4,2),
                heart_rate INT,
                bp_sys INT,
                bp_dia INT,
                oxygen_saturation DECIMAL(4,2),
                respiratory_rate INT,
                glucose_level DECIMAL(5,2),
                
                -- Symptoms (1 for present, 0 for absent)
                fever BOOLEAN DEFAULT FALSE,
                cough BOOLEAN DEFAULT FALSE,
                chest_pain BOOLEAN DEFAULT FALSE,
                shortness_of_breath BOOLEAN DEFAULT FALSE,
                fatigue BOOLEAN DEFAULT FALSE,
                headache BOOLEAN DEFAULT FALSE,
                nausea BOOLEAN DEFAULT FALSE,
                dizziness BOOLEAN DEFAULT FALSE,
                
                -- Environmental
                humidity DECIMAL(5,2),
                temperature_environment DECIMAL(4,2),
                
                -- Prediction Results
                predicted_disease VARCHAR(100),
                confidence DECIMAL(5,4),
                probabilities JSON,
                risk_level ENUM('Low', 'Medium', 'High', 'Critical'),
                
                -- Additional Info
                notes TEXT,
                doctor_recommendations TEXT,
                follow_up_date DATE,
                
                record_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                INDEX idx_patient_date (patient_id, record_date),
                INDEX idx_disease (predicted_disease),
                INDEX idx_risk_level (risk_level),
                INDEX idx_visit_number (patient_id, visit_number),
                INDEX idx_record_date (record_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Patient statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_stats (
                stat_id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id VARCHAR(50),
                stat_date DATE,
                
                -- Averages
                avg_temperature DECIMAL(4,2),
                avg_heart_rate DECIMAL(5,2),
                avg_bp_sys DECIMAL(5,2),
                avg_bp_dia DECIMAL(5,2),
                avg_glucose DECIMAL(5,2),
                
                -- Symptom frequencies
                fever_count INT DEFAULT 0,
                cough_count INT DEFAULT 0,
                chest_pain_count INT DEFAULT 0,
                sob_count INT DEFAULT 0,
                fatigue_count INT DEFAULT 0,
                headache_count INT DEFAULT 0,
                
                -- Disease history
                disease_history JSON,
                most_common_disease VARCHAR(100),
                disease_frequency INT,
                
                -- Risk analysis
                risk_trend ENUM('Improving', 'Stable', 'Worsening'),
                severity_score INT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                UNIQUE KEY unique_patient_date (patient_id, stat_date),
                INDEX idx_stat_date (stat_date),
                INDEX idx_risk_trend (risk_trend)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Disease patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS disease_patterns (
                pattern_id INT AUTO_INCREMENT PRIMARY KEY,
                disease_name VARCHAR(100),
                common_symptoms JSON,
                avg_temperature_range VARCHAR(20),
                avg_heart_rate_range VARCHAR(20),
                avg_bp_range VARCHAR(20),
                seasonality VARCHAR(50),
                risk_factors JSON,
                prevention_measures JSON,
                total_cases INT DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                UNIQUE KEY unique_disease (disease_name),
                INDEX idx_disease_name (disease_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id INT AUTO_INCREMENT PRIMARY KEY,
                action_type VARCHAR(50),
                table_name VARCHAR(50),
                record_id VARCHAR(100),
                old_values JSON,
                new_values JSON,
                performed_by VARCHAR(100),
                performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                INDEX idx_action_type (action_type),
                INDEX idx_performed_at (performed_at),
                INDEX idx_table_record (table_name, record_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        logger.info("✅ All tables created successfully")
    
    # Patient Management Methods
    
    def create_patient(self, patient_data: Dict) -> Optional[str]:
        """Create a new patient record"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor()
                    
                    # Generate patient ID if not provided
                    if 'patient_id' not in patient_data or not patient_data['patient_id']:
                        patient_data['patient_id'] = f"PAT{uuid.uuid4().hex[:8].upper()}"
                    
                    query = """
                        INSERT INTO patients (
                            patient_id, first_name, last_name, age, gender,
                            contact_number, email, address, emergency_contact,
                            blood_group, allergies, medical_history
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    values = (
                        patient_data['patient_id'],
                        patient_data.get('first_name'),
                        patient_data.get('last_name'),
                        patient_data.get('age'),
                        patient_data.get('gender'),
                        patient_data.get('contact_number'),
                        patient_data.get('email'),
                        patient_data.get('address'),
                        patient_data.get('emergency_contact'),
                        patient_data.get('blood_group'),
                        patient_data.get('allergies'),
                        patient_data.get('medical_history')
                    )
                    
                    cursor.execute(query, values)
                    connection.commit()
                    
                    # Log the action
                    self._log_audit('INSERT', 'patients', patient_data['patient_id'], 
                                   None, patient_data, 'system')
                    
                    logger.info(f"✅ Patient created: {patient_data['patient_id']}")
                    return patient_data['patient_id']
                    
        except Error as e:
            logger.error(f"❌ Error creating patient: {e}")
            return None
    
    def update_patient(self, patient_id: str, update_data: Dict) -> bool:
        """Update patient information"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor(dictionary=True)
                    
                    # Get old values for audit
                    cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
                    old_values = cursor.fetchone()
                    
                    # Build update query
                    set_clause = []
                    values = []
                    for key, value in update_data.items():
                        if key != 'patient_id' and value is not None:
                            set_clause.append(f"{key} = %s")
                            values.append(value)
                    
                    if not set_clause:
                        return False
                    
                    values.append(patient_id)
                    query = f"UPDATE patients SET {', '.join(set_clause)} WHERE patient_id = %s"
                    
                    cursor.execute(query, values)
                    connection.commit()
                    
                    # Log the action
                    self._log_audit('UPDATE', 'patients', patient_id, 
                                   old_values, update_data, 'system')
                    
                    logger.info(f"✅ Patient updated: {patient_id}")
                    return True
                    
        except Error as e:
            logger.error(f"❌ Error updating patient: {e}")
            return False
    
    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Get patient details"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
                    patient = cursor.fetchone()
                    
                    if patient:
                        # Get latest health record
                        cursor.execute("""
                            SELECT predicted_disease, confidence, record_date 
                            FROM health_records 
                            WHERE patient_id = %s 
                            ORDER BY record_date DESC 
                            LIMIT 1
                        """, (patient_id,))
                        latest_record = cursor.fetchone()
                        
                        if latest_record:
                            patient['latest_disease'] = latest_record['predicted_disease']
                            patient['latest_confidence'] = latest_record['confidence']
                            patient['last_checkup'] = latest_record['record_date']
                        
                        # Get visit statistics
                        cursor.execute("""
                            SELECT 
                                COUNT(*) as total_records,
                                MIN(record_date) as first_visit,
                                MAX(record_date) as last_visit
                            FROM health_records 
                            WHERE patient_id = %s
                        """, (patient_id,))
                        stats = cursor.fetchone()
                        
                        if stats:
                            patient.update(stats)
                    
                    return patient
                    
        except Error as e:
            logger.error(f"❌ Error getting patient: {e}")
            return None
    
    # Health Records Methods
    
    def save_health_record(self, record_data: Dict) -> Optional[int]:
        """Save a new health record"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor()
                    
                    # Get visit number
                    cursor.execute("""
                        SELECT COALESCE(MAX(visit_number), 0) + 1 as next_visit
                        FROM health_records 
                        WHERE patient_id = %s
                    """, (record_data['patient_id'],))
                    result = cursor.fetchone()
                    visit_number = result[0] if result else 1
                    
                    # Calculate risk level based on symptoms and vitals
                    risk_level = self._calculate_risk_level(record_data)
                    
                    query = """
                        INSERT INTO health_records (
                            patient_id, visit_number,
                            temperature, heart_rate, bp_sys, bp_dia, 
                            oxygen_saturation, respiratory_rate, glucose_level,
                            fever, cough, chest_pain, shortness_of_breath,
                            fatigue, headache, nausea, dizziness,
                            humidity, temperature_environment,
                            predicted_disease, confidence, probabilities,
                            risk_level, notes, doctor_recommendations, follow_up_date
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                                 %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    values = (
                        record_data['patient_id'],
                        visit_number,
                        record_data.get('temperature'),
                        record_data.get('heart_rate'),
                        record_data.get('bp_sys'),
                        record_data.get('bp_dia'),
                        record_data.get('oxygen_saturation'),
                        record_data.get('respiratory_rate'),
                        record_data.get('glucose_level'),
                        record_data.get('fever', False),
                        record_data.get('cough', False),
                        record_data.get('chest_pain', False),
                        record_data.get('shortness_of_breath', False),
                        record_data.get('fatigue', False),
                        record_data.get('headache', False),
                        record_data.get('nausea', False),
                        record_data.get('dizziness', False),
                        record_data.get('humidity'),
                        record_data.get('temperature_environment'),
                        record_data.get('predicted_disease'),
                        record_data.get('confidence'),
                        json.dumps(record_data.get('probabilities', {})),
                        risk_level,
                        record_data.get('notes'),
                        record_data.get('doctor_recommendations'),
                        record_data.get('follow_up_date')
                    )
                    
                    cursor.execute(query, values)
                    record_id = cursor.lastrowid
                    
                    # Update patient's visit count
                    cursor.execute("""
                        UPDATE patients 
                        SET total_visits = total_visits + 1,
                            last_visit = CURRENT_TIMESTAMP
                        WHERE patient_id = %s
                    """, (record_data['patient_id'],))
                    
                    # Update patient statistics
                    self._update_patient_stats(record_data['patient_id'])
                    
                    connection.commit()
                    
                    # Log the action
                    self._log_audit('INSERT', 'health_records', record_id, 
                                   None, record_data, 'system')
                    
                    logger.info(f"✅ Health record saved: ID {record_id} for patient {record_data['patient_id']}")
                    return record_id
                    
        except Error as e:
            logger.error(f"❌ Error saving health record: {e}")
            return None
    
    def get_health_records(self, patient_id: str, limit: int = 50, 
                          start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get health records for a patient"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor(dictionary=True)
                    
                    query = """
                        SELECT 
                            record_id, visit_number,
                            temperature, heart_rate, bp_sys, bp_dia,
                            oxygen_saturation, respiratory_rate, glucose_level,
                            fever, cough, chest_pain, shortness_of_breath,
                            fatigue, headache, nausea, dizziness,
                            humidity, temperature_environment,
                            predicted_disease, confidence, probabilities,
                            risk_level, notes, doctor_recommendations, follow_up_date,
                            DATE_FORMAT(record_date, '%%Y-%%m-%%d %%H:%%i:%%s') as record_date
                        FROM health_records 
                        WHERE patient_id = %s
                    """
                    
                    params = [patient_id]
                    
                    # FIXED: Properly handle optional date parameters
                    if start_date and end_date:
                        query += " AND DATE(record_date) BETWEEN %s AND %s"
                        params.extend([start_date, end_date])
                    elif start_date:
                        query += " AND DATE(record_date) >= %s"
                        params.append(start_date)
                    elif end_date:
                        query += " AND DATE(record_date) <= %s"
                        params.append(end_date)
                    
                    query += " ORDER BY record_date DESC LIMIT %s"
                    params.append(limit)
                    
                    cursor.execute(query, tuple(params))  # Convert list to tuple
                    records = cursor.fetchall()
                    
                    # Convert JSON strings back to objects
                    for record in records:
                        if record.get('probabilities'):
                            try:
                                record['probabilities'] = json.loads(record['probabilities'])
                            except:
                                record['probabilities'] = {}
                    
                    return records
                    
        except Error as e:
            logger.error(f"❌ Error getting health records: {e}")
            return []
    
    def get_patient_history(self, patient_id: str) -> List[Dict]:
        """Get patient history (simplified version of get_health_records)"""
        try:
            return self.get_health_records(patient_id, limit=100)
        except Error as e:
            logger.error(f"❌ Error fetching patient history: {e}")
            return []
    
    # Analytics Methods
    
    def get_patient_analysis(self, patient_id: str, days: int = 30) -> Dict:
        """Get comprehensive analysis for a patient"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor(dictionary=True)
                    
                    # Get symptom frequency data
                    cursor.execute("""
                        SELECT 
                            DATE(record_date) as record_date,
                            COUNT(*) as total_checks,
                            SUM(fever) as fever_count,
                            SUM(cough) as cough_count,
                            SUM(chest_pain) as chest_pain_count,
                            SUM(shortness_of_breath) as sob_count,
                            SUM(fatigue) as fatigue_count,
                            SUM(headache) as headache_count,
                            SUM(nausea) as nausea_count,
                            SUM(dizziness) as dizziness_count,
                            GROUP_CONCAT(DISTINCT predicted_disease) as diseases
                        FROM health_records 
                        WHERE patient_id = %s 
                          AND record_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                        GROUP BY DATE(record_date)
                        ORDER BY record_date ASC
                    """, (patient_id, days))
                    
                    symptom_data = cursor.fetchall()
                    
                    # Get vital signs trend
                    cursor.execute("""
                        SELECT 
                            DATE(record_date) as record_date,
                            AVG(temperature) as avg_temp,
                            AVG(heart_rate) as avg_hr,
                            AVG(bp_sys) as avg_sys,
                            AVG(bp_dia) as avg_dia,
                            AVG(oxygen_saturation) as avg_spo2,
                            AVG(glucose_level) as avg_glucose,
                            COUNT(*) as records_count
                        FROM health_records 
                        WHERE patient_id = %s 
                          AND record_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                        GROUP BY DATE(record_date)
                        ORDER BY record_date ASC
                    """, (patient_id, days))
                    
                    vital_signs = cursor.fetchall()
                    
                    # Get disease progression
                    cursor.execute("""
                        SELECT 
                            predicted_disease,
                            COUNT(*) as frequency,
                            AVG(confidence) as avg_confidence,
                            MIN(record_date) as first_occurrence,
                            MAX(record_date) as last_occurrence
                        FROM health_records 
                        WHERE patient_id = %s
                        GROUP BY predicted_disease
                        ORDER BY frequency DESC
                    """, (patient_id,))
                    
                    disease_progression = cursor.fetchall()
                    
                    # Calculate risk trends
                    cursor.execute("""
                        SELECT risk_level 
                        FROM health_records 
                        WHERE patient_id = %s 
                        ORDER BY record_date DESC 
                        LIMIT 2
                    """, (patient_id,))
                    
                    recent_risks = [row['risk_level'] for row in cursor.fetchall()]
                    risk_trend = 'Stable'
                    
                    if len(recent_risks) == 2:
                        risk_order = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
                        risk1 = risk_order.get(recent_risks[0], 0)
                        risk2 = risk_order.get(recent_risks[1], 0)
                        
                        if risk1 > risk2:
                            risk_trend = 'Worsening'
                        elif risk1 < risk2:
                            risk_trend = 'Improving'
                    
                    return {
                        "symptom_frequency": symptom_data,
                        "vital_signs_trend": vital_signs,
                        "disease_progression": disease_progression,
                        "risk_analysis": {"risk_trend": risk_trend},
                        "analysis_period": f"{days} days",
                        "generated_at": datetime.now().isoformat()
                    }
                    
        except Error as e:
            logger.error(f"❌ Error getting patient analysis: {e}")
            return {}
    
    def get_system_statistics(self) -> Dict:
        """Get overall system statistics"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor(dictionary=True)
                    
                    # Basic counts
                    cursor.execute("SELECT COUNT(*) as total_patients FROM patients")
                    result = cursor.fetchone()
                    total_patients = result['total_patients'] if result else 0
                    
                    cursor.execute("SELECT COUNT(*) as total_records FROM health_records")
                    result = cursor.fetchone()
                    total_records = result['total_records'] if result else 0
                    
                    # Today's activity
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as today_records,
                            COUNT(DISTINCT patient_id) as today_patients
                        FROM health_records 
                        WHERE DATE(record_date) = CURDATE()
                    """)
                    today_activity = cursor.fetchone() or {"today_records": 0, "today_patients": 0}
                    
                    # Disease distribution
                    cursor.execute("""
                        SELECT 
                            predicted_disease,
                            COUNT(*) as count,
                            ROUND(COUNT(*) * 100.0 / (SELECT GREATEST(COUNT(*), 1) FROM health_records), 2) as percentage
                        FROM health_records 
                        WHERE predicted_disease IS NOT NULL
                        GROUP BY predicted_disease
                        ORDER BY count DESC
                    """)
                    disease_distribution = cursor.fetchall()
                    
                    # Risk level distribution
                    cursor.execute("""
                        SELECT 
                            risk_level,
                            COUNT(*) as count
                        FROM health_records 
                        WHERE risk_level IS NOT NULL
                        GROUP BY risk_level
                        ORDER BY FIELD(risk_level, 'Critical', 'High', 'Medium', 'Low')
                    """)
                    risk_distribution = cursor.fetchall()
                    
                    # Patient demographics
                    cursor.execute("""
                        SELECT 
                            gender,
                            COUNT(*) as count,
                            ROUND(AVG(age)) as avg_age
                        FROM patients 
                        WHERE gender IS NOT NULL
                        GROUP BY gender
                    """)
                    demographics = cursor.fetchall()
                    
                    return {
                        "total_patients": total_patients,
                        "total_records": total_records,
                        "today_activity": today_activity,
                        "disease_distribution": disease_distribution,
                        "risk_distribution": risk_distribution,
                        "demographics": demographics,
                        "generated_at": datetime.now().isoformat()
                    }
                    
        except Error as e:
            logger.error(f"❌ Error getting system statistics: {e}")
            return {}
    
    # Search Methods
    
    def search_patients(self, search_term: str = None, filters: Dict = None, 
                       limit: int = 50, offset: int = 0) -> Dict:
        """Search for patients with filters"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor(dictionary=True)
                    
                    base_query = """
                        SELECT 
                            p.*,
                            hr.predicted_disease as latest_disease,
                            hr.confidence as latest_confidence,
                            hr.record_date as last_checkup,
                            hr.risk_level as latest_risk
                        FROM patients p
                        LEFT JOIN health_records hr ON p.patient_id = hr.patient_id
                        WHERE 1=1
                    """
                    
                    params = []
                    
                    # Search term
                    if search_term:
                        base_query += """
                            AND (
                                p.patient_id LIKE %s OR
                                p.first_name LIKE %s OR
                                p.last_name LIKE %s OR
                                p.contact_number LIKE %s OR
                                p.email LIKE %s
                            )
                        """
                        search_pattern = f"%{search_term}%"
                        params.extend([search_pattern] * 5)
                    
                    # Additional filters
                    if filters:
                        if filters.get('gender'):
                            base_query += " AND p.gender = %s"
                            params.append(filters['gender'])
                        
                        if filters.get('min_age'):
                            base_query += " AND p.age >= %s"
                            params.append(filters['min_age'])
                        
                        if filters.get('max_age'):
                            base_query += " AND p.age <= %s"
                            params.append(filters['max_age'])
                        
                        if filters.get('disease'):
                            base_query += """
                                AND p.patient_id IN (
                                    SELECT patient_id 
                                    FROM health_records 
                                    WHERE predicted_disease = %s
                                )
                            """
                            params.append(filters['disease'])
                    
                    # Get total count
                    count_query = f"SELECT COUNT(*) as total FROM ({base_query}) as filtered"
                    cursor.execute(count_query, params)
                    result = cursor.fetchone()
                    total_count = result['total'] if result else 0
                    
                    # Get paginated results
                    base_query += """
                        GROUP BY p.patient_id
                        ORDER BY p.last_visit DESC
                        LIMIT %s OFFSET %s
                    """
                    params.extend([limit, offset])
                    
                    cursor.execute(base_query, tuple(params))
                    patients = cursor.fetchall()
                    
                    return {
                        "patients": patients,
                        "total_count": total_count,
                        "current_page": offset // limit + 1 if limit > 0 else 1,
                        "total_pages": (total_count + limit - 1) // limit if limit > 0 else 1,
                        "limit": limit,
                        "offset": offset
                    }
                    
        except Error as e:
            logger.error(f"❌ Error searching patients: {e}")
            return {"patients": [], "total_count": 0}
    
    # Helper Methods
    
    def _calculate_risk_level(self, record_data: Dict) -> str:
        """Calculate risk level based on symptoms and vitals"""
        risk_score = 0
        
        # Check critical vitals
        if record_data.get('temperature', 0) > 39.0:
            risk_score += 2
        if record_data.get('heart_rate', 0) > 120 or record_data.get('heart_rate', 0) < 50:
            risk_score += 2
        if record_data.get('bp_sys', 0) > 180 or record_data.get('bp_dia', 0) > 120:
            risk_score += 3
        if record_data.get('bp_sys', 0) < 90 or record_data.get('bp_dia', 0) < 60:
            risk_score += 2
        if record_data.get('oxygen_saturation', 0) < 92:
            risk_score += 3
        
        # Check symptoms
        critical_symptoms = ['chest_pain', 'shortness_of_breath']
        for symptom in critical_symptoms:
            if record_data.get(symptom):
                risk_score += 2
        
        other_symptoms = ['fever', 'cough', 'fatigue', 'headache', 'nausea', 'dizziness']
        for symptom in other_symptoms:
            if record_data.get(symptom):
                risk_score += 1
        
        # Determine risk level
        if risk_score >= 8:
            return 'Critical'
        elif risk_score >= 5:
            return 'High'
        elif risk_score >= 3:
            return 'Medium'
        else:
            return 'Low'
    
    def _update_patient_stats(self, patient_id: str):
        """Update patient statistics"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor(dictionary=True)
                    
                    # FIXED: Use CURDATE() for stat_date instead of querying date column
                    cursor.execute("""
                        SELECT 
                            CURDATE() as stat_date,
                            AVG(temperature) as avg_temp,
                            AVG(heart_rate) as avg_hr,
                            AVG(bp_sys) as avg_sys,
                            AVG(bp_dia) as avg_dia,
                            AVG(glucose_level) as avg_glucose,
                            SUM(fever) as fever_count,
                            SUM(cough) as cough_count,
                            SUM(chest_pain) as chest_pain_count,
                            SUM(shortness_of_breath) as sob_count,
                            SUM(fatigue) as fatigue_count,
                            SUM(headache) as headache_count,
                            GROUP_CONCAT(predicted_disease) as diseases
                        FROM health_records 
                        WHERE patient_id = %s 
                          AND DATE(record_date) = CURDATE()
                        GROUP BY DATE(record_date)
                    """, (patient_id,))
                    
                    stats = cursor.fetchone()
                    
                    if stats and stats['stat_date']:
                        disease_list = stats['diseases'].split(',') if stats['diseases'] else []
                        disease_count = {}
                        for disease in disease_list:
                            if disease:  # Skip empty strings
                                disease_count[disease] = disease_count.get(disease, 0) + 1
                        
                        most_common = max(disease_count.items(), key=lambda x: x[1]) if disease_count else (None, 0)
                        
                        # Calculate severity score
                        severity_score = (
                            (stats['fever_count'] or 0) * 1 +
                            (stats['cough_count'] or 0) * 1 +
                            (stats['chest_pain_count'] or 0) * 3 +
                            (stats['sob_count'] or 0) * 2 +
                            (stats['fatigue_count'] or 0) * 1 +
                            (stats['headache_count'] or 0) * 1
                        )
                        
                        # Determine risk trend
                        cursor.execute("""
                            SELECT risk_level 
                            FROM health_records 
                            WHERE patient_id = %s 
                            ORDER BY record_date DESC 
                            LIMIT 2
                        """, (patient_id,))
                        
                        rows = cursor.fetchall()
                        recent_risks = [row['risk_level'] for row in rows] if rows else []
                        risk_trend = 'Stable'
                        
                        if len(recent_risks) == 2:
                            risk_order = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
                            risk1 = risk_order.get(recent_risks[0], 0)
                            risk2 = risk_order.get(recent_risks[1], 0)
                            
                            if risk1 > risk2:
                                risk_trend = 'Worsening'
                            elif risk1 < risk2:
                                risk_trend = 'Improving'
                        
                        # Insert or update stats
                        query = """
                            INSERT INTO patient_stats (
                                patient_id, stat_date,
                                avg_temperature, avg_heart_rate, avg_bp_sys, avg_bp_dia, avg_glucose,
                                fever_count, cough_count, chest_pain_count, sob_count,
                                fatigue_count, headache_count,
                                disease_history, most_common_disease, disease_frequency,
                                risk_trend, severity_score
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                                avg_temperature = VALUES(avg_temperature),
                                avg_heart_rate = VALUES(avg_heart_rate),
                                avg_bp_sys = VALUES(avg_bp_sys),
                                avg_bp_dia = VALUES(avg_bp_dia),
                                avg_glucose = VALUES(avg_glucose),
                                fever_count = VALUES(fever_count),
                                cough_count = VALUES(cough_count),
                                chest_pain_count = VALUES(chest_pain_count),
                                sob_count = VALUES(sob_count),
                                fatigue_count = VALUES(fatigue_count),
                                headache_count = VALUES(headache_count),
                                disease_history = VALUES(disease_history),
                                most_common_disease = VALUES(most_common_disease),
                                disease_frequency = VALUES(disease_frequency),
                                risk_trend = VALUES(risk_trend),
                                severity_score = VALUES(severity_score)
                        """
                        
                        values = (
                            patient_id,
                            stats['stat_date'],
                            stats['avg_temp'] or 0,
                            stats['avg_hr'] or 0,
                            stats['avg_sys'] or 0,
                            stats['avg_dia'] or 0,
                            stats['avg_glucose'] or 0,
                            stats['fever_count'] or 0,
                            stats['cough_count'] or 0,
                            stats['chest_pain_count'] or 0,
                            stats['sob_count'] or 0,
                            stats['fatigue_count'] or 0,
                            stats['headache_count'] or 0,
                            json.dumps(disease_count),
                            most_common[0],
                            most_common[1],
                            risk_trend,
                            severity_score
                        )
                        
                        cursor.execute(query, values)
                        connection.commit()
                        logger.info(f"✅ Patient stats updated for {patient_id}")
                        
        except Error as e:
            logger.error(f"❌ Error updating patient stats: {e}")
    
    def _log_audit(self, action_type: str, table_name: str, record_id: Any, 
                  old_values: Dict, new_values: Dict, performed_by: str):
        """Log audit trail"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor()
                    
                    query = """
                        INSERT INTO audit_log 
                        (action_type, table_name, record_id, old_values, new_values, performed_by)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    
                    cursor.execute(query, (
                        action_type,
                        table_name,
                        str(record_id),
                        json.dumps(old_values) if old_values else None,
                        json.dumps(new_values) if new_values else None,
                        performed_by
                    ))
                    
                    connection.commit()
                    
        except Error as e:
            logger.error(f"❌ Error logging audit: {e}")
    
    # Backup and Maintenance Methods
    
    def backup_database(self, backup_path: str = "backups") -> bool:
        """Create database backup"""
        try:
            import subprocess
            import time
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_path, f"health_db_backup_{timestamp}.sql")
            
            # Create backup directory if not exists
            os.makedirs(backup_path, exist_ok=True)
            
            # Create backup command
            cmd = [
                "mysqldump",
                "-h", self.config.host,
                "-u", self.config.user,
                f"-p{self.config.password}",
                self.config.database
            ]
            
            with open(backup_file, 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)
            
            logger.info(f"✅ Database backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database backup failed: {e}")
            return False
    
    def optimize_tables(self) -> bool:
        """Optimize database tables"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor()
                    
                    # Get all tables
                    cursor.execute("SHOW TABLES")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    for table in tables:
                        cursor.execute(f"OPTIMIZE TABLE {table}")
                    
                    connection.commit()
                    logger.info("✅ Database tables optimized")
                    return True
                    
        except Error as e:
            logger.error(f"❌ Table optimization failed: {e}")
            return False

# Singleton instance
db_manager = DatabaseManager()

# Test function
def test_database_connection():
    """Test database connection"""
    try:
        connection = db_manager.get_connection()
        if connection and connection.is_connected():
            logger.info("✅ Database connection test: PASSED")
            connection.close()
            return True
        else:
            logger.error("❌ Database connection test: FAILED")
            return False
    except Exception as e:
        logger.error(f"❌ Database test error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print(" HEALTH PREDICTION DATABASE MANAGER")
    print("="*60)
    
    if test_database_connection():
        print("✅ Database is ready for use")
        
        # Test creating a sample patient
        sample_patient = {
            "first_name": "John",
            "last_name": "Doe",
            "age": 35,
            "gender": "Male",
            "contact_number": "1234567890",
            "blood_group": "O+"
        }
        
        patient_id = db_manager.create_patient(sample_patient)
        if patient_id:
            print(f"✅ Sample patient created: {patient_id}")
            
            # Test saving a health record
            sample_record = {
                "patient_id": patient_id,
                "temperature": 37.5,
                "heart_rate": 85,
                "bp_sys": 120,
                "bp_dia": 80,
                "fever": True,
                "cough": False,
                "predicted_disease": "Fever_Respiratory",
                "confidence": 0.85
            }
            
            record_id = db_manager.save_health_record(sample_record)
            if record_id:
                print(f"✅ Sample health record saved: {record_id}")
                
                # Test getting patient analysis
                analysis = db_manager.get_patient_analysis(patient_id, days=7)
                if analysis:
                    print(f"✅ Patient analysis generated")
                    print(f"   Symptom frequency data: {len(analysis['symptom_frequency'])} days")
                    print(f"   Vital signs trend: {len(analysis['vital_signs_trend'])} days")
        
        # Test system statistics
        stats = db_manager.get_system_statistics()
        if stats:
            print(f"✅ System statistics retrieved")
            print(f"   Total patients: {stats['total_patients']}")
            print(f"   Total records: {stats['total_records']}")
    
    print("="*60)