import mysql.connector
from mysql.connector import Error
import os
from datetime import datetime
import json
import logging
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
class DatabaseConfig:
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.user = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASSWORD", "1234")
        self.database = os.getenv("DB_NAME", "health_prediction_system")
        self.port = int(os.getenv("DB_PORT", 3306))
        self.pool_name = "health_pool"
        self.pool_size = 10
        
    def get_connection_params(self):
        return {
            "host": self.host,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "port": self.port
        }

class DatabaseManager:
    """Main database manager for health prediction system"""
    
    def __init__(self):
        self.config = DatabaseConfig()
        self.connection_pool = None
        self.init_database()
    
    def get_connection(self):
        """Get a database connection"""
        try:
            return mysql.connector.connect(
                host=self.config.host,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                port=self.config.port
            )
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
        
        # Patients table with expanded fields
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id VARCHAR(50) PRIMARY KEY,
                first_name VARCHAR(100),
                last_name VARCHAR(100),
                age INT,
                gender VARCHAR(10),
                contact_number VARCHAR(20),
                email VARCHAR(100),
                blood_group VARCHAR(5),
                allergies TEXT,
                medical_history TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_visit TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                total_visits INT DEFAULT 0,
                INDEX idx_last_visit (last_visit),
                INDEX idx_total_visits (total_visits),
                INDEX idx_name (first_name, last_name),
                INDEX idx_contact (contact_number)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        
        # Health records table (simplified from previous version)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_records (
                record_id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id VARCHAR(50),
                
                -- Vital Signs
                temperature DECIMAL(4,2),
                heart_rate INT,
                bp_sys INT,
                bp_dia INT,
                humidity DECIMAL(5,2),
                
                -- Symptoms
                fever BOOLEAN DEFAULT FALSE,
                cough BOOLEAN DEFAULT FALSE,
                chest_pain BOOLEAN DEFAULT FALSE,
                shortness_of_breath BOOLEAN DEFAULT FALSE,
                fatigue BOOLEAN DEFAULT FALSE,
                headache BOOLEAN DEFAULT FALSE,
                
                -- Prediction Results
                predicted_disease VARCHAR(100),
                confidence DECIMAL(5,4),
                
                record_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
                INDEX idx_patient_date (patient_id, record_date),
                INDEX idx_disease (predicted_disease)
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
                            contact_number, email, blood_group, allergies, medical_history
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    values = (
                        patient_data['patient_id'],
                        patient_data.get('first_name'),
                        patient_data.get('last_name'),
                        patient_data.get('age'),
                        patient_data.get('gender'),
                        patient_data.get('contact_number'),
                        patient_data.get('email'),
                        patient_data.get('blood_group'),
                        patient_data.get('allergies'),
                        patient_data.get('medical_history')
                    )
                    
                    cursor.execute(query, values)
                    connection.commit()
                    
                    logger.info(f"✅ Patient created: {patient_data['patient_id']}")
                    return patient_data['patient_id']
                    
        except Error as e:
            logger.error(f"❌ Error creating patient: {e}")
            return None
    
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
                        
                        # Get total records count
                        cursor.execute("""
                            SELECT COUNT(*) as total_records 
                            FROM health_records 
                            WHERE patient_id = %s
                        """, (patient_id,))
                        count_result = cursor.fetchone()
                        patient['total_records'] = count_result['total_records'] if count_result else 0
                    
                    cursor.close()
                    return patient
                    
        except Error as e:
            logger.error(f"❌ Error getting patient: {e}")
            return None
    
    def search_patients(self, search_term: str = None, limit: int = 20, offset: int = 0) -> Dict:
        """Search for patients"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor(dictionary=True)
                    
                    base_query = """
                        SELECT 
                            p.*,
                            COUNT(hr.record_id) as total_visits,
                            MAX(hr.record_date) as last_visit
                        FROM patients p
                        LEFT JOIN health_records hr ON p.patient_id = hr.patient_id
                        WHERE 1=1
                    """
                    
                    params = []
                    
                    if search_term:
                        base_query += """
                            AND (
                                p.patient_id LIKE %s OR
                                p.first_name LIKE %s OR
                                p.last_name LIKE %s OR
                                CONCAT(p.first_name, ' ', p.last_name) LIKE %s OR
                                p.contact_number LIKE %s OR
                                p.email LIKE %s
                            )
                        """
                        search_pattern = f"%{search_term}%"
                        params.extend([search_pattern] * 6)
                    
                    base_query += """
                        GROUP BY p.patient_id
                        ORDER BY p.created_at DESC
                        LIMIT %s OFFSET %s
                    """
                    params.extend([limit, offset])
                    
                    cursor.execute(base_query, tuple(params))
                    patients = cursor.fetchall()
                    
                    # Get total count for pagination
                    count_query = """
                        SELECT COUNT(DISTINCT p.patient_id) as total
                        FROM patients p
                        WHERE 1=1
                    """
                    count_params = []
                    
                    if search_term:
                        count_query += """
                            AND (
                                p.patient_id LIKE %s OR
                                p.first_name LIKE %s OR
                                p.last_name LIKE %s OR
                                CONCAT(p.first_name, ' ', p.last_name) LIKE %s OR
                                p.contact_number LIKE %s OR
                                p.email LIKE %s
                            )
                        """
                        count_params.extend([search_pattern] * 6)
                    
                    cursor.execute(count_query, tuple(count_params))
                    total = cursor.fetchone()['total']
                    
                    cursor.close()
                    
                    return {
                        "patients": patients,
                        "total": total,
                        "limit": limit,
                        "offset": offset,
                        "has_more": (offset + len(patients)) < total
                    }
                    
        except Error as e:
            logger.error(f"❌ Error searching patients: {e}")
            return {"patients": [], "total": 0}
    
    # Health Records Methods
    
    def create_or_update_patient(self, patient_id: str) -> bool:
        """Create or update patient for predictions"""
        try:
            with self.get_db_connection() as conn:
                if not conn:
                    return False
                cursor = conn.cursor()
                
                # Check if patient exists
                cursor.execute("SELECT patient_id FROM patients WHERE patient_id = %s", (patient_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing patient
                    cursor.execute("""
                        UPDATE patients 
                        SET total_visits = total_visits + 1, 
                            last_visit = CURRENT_TIMESTAMP 
                        WHERE patient_id = %s
                    """, (patient_id,))
                else:
                    # Create new minimal patient record
                    cursor.execute("""
                        INSERT INTO patients (patient_id, total_visits) 
                        VALUES (%s, 1)
                    """, (patient_id,))
                
                conn.commit()
                cursor.close()
                return True
                
        except Error as e:
            logger.error(f"Error creating/updating patient: {e}")
            return False
    
    def save_health_record(self, patient_id: str, features: dict, prediction_result: dict) -> Optional[int]:
        """Save a health record"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor()
                    
                    # Ensure patient exists
                    self.create_or_update_patient(patient_id)
                    
                    query = """
                        INSERT INTO health_records (
                            patient_id, temperature, heart_rate, bp_sys, bp_dia, humidity,
                            fever, cough, chest_pain, shortness_of_breath, fatigue, headache,
                            predicted_disease, confidence
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    values = (
                        patient_id,
                        features.get('temperature'),
                        features.get('heart_rate'),
                        features.get('bp_sys'),
                        features.get('bp_dia'),
                        features.get('humidity'),
                        bool(features.get('fever', 0)),
                        bool(features.get('cough', 0)),
                        bool(features.get('chest_pain', 0)),
                        bool(features.get('shortness_of_breath', 0)),
                        bool(features.get('fatigue', 0)),
                        bool(features.get('headache', 0)),
                        prediction_result.get('disease'),
                        prediction_result.get('confidence')
                    )
                    
                    cursor.execute(query, values)
                    record_id = cursor.lastrowid
                    
                    connection.commit()
                    cursor.close()
                    
                    logger.info(f"✅ Health record saved: ID {record_id} for patient {patient_id}")
                    return record_id
                    
        except Error as e:
            logger.error(f"❌ Error saving health record: {e}")
            return None
    
    def get_patient_history(self, patient_id: str, limit: int = 50) -> Dict:
        """Get patient health history"""
        try:
            with self.get_db_connection() as connection:
                if connection:
                    cursor = connection.cursor(dictionary=True)
                    
                    # Get health records
                    cursor.execute("""
                        SELECT 
                            record_id, temperature, heart_rate, bp_sys, bp_dia, humidity,
                            fever, cough, chest_pain, shortness_of_breath, fatigue, headache,
                            predicted_disease, confidence,
                            DATE_FORMAT(record_date, '%%Y-%%m-%%d %%H:%%i:%%s') as record_date
                        FROM health_records 
                        WHERE patient_id = %s 
                        ORDER BY record_date DESC 
                        LIMIT %s
                    """, (patient_id, limit))
                    
                    records = cursor.fetchall()
                    
                    # Get patient info
                    cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
                    patient_info = cursor.fetchone()
                    
                    cursor.close()
                    
                    return {
                        "patient_info": patient_info,
                        "health_records": records,
                        "total_records": len(records)
                    }
                    
        except Error as e:
            logger.error(f"❌ Error fetching patient history: {e}")
            return None
    
    # Statistics Methods
    
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
                    
                    cursor.close()
                    
                    return {
                        "total_patients": total_patients,
                        "total_records": total_records,
                        "today_activity": today_activity,
                        "disease_distribution": disease_distribution,
                        "generated_at": datetime.now().isoformat()
                    }
                    
        except Error as e:
            logger.error(f"❌ Error getting system statistics: {e}")
            return {}
    
    # Maintenance Methods
    
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
            "blood_group": "O+",
            "allergies": "Penicillin"
        }
        
        patient_id = db_manager.create_patient(sample_patient)
        if patient_id:
            print(f"✅ Sample patient created: {patient_id}")
            
            # Test getting patient
            patient = db_manager.get_patient(patient_id)
            if patient:
                print(f"✅ Patient retrieved: {patient['first_name']} {patient['last_name']}")
            
            # Test searching patients
            search_result = db_manager.search_patients("John", limit=5)
            if search_result['patients']:
                print(f"✅ Search found {len(search_result['patients'])} patients")
        
        # Test system statistics
        stats = db_manager.get_system_statistics()
        if stats:
            print(f"✅ System statistics retrieved")
            print(f"   Total patients: {stats['total_patients']}")
            print(f"   Total records: {stats['total_records']}")
    
    print("="*60)