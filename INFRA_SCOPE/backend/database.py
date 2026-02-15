import sqlite3
import json
import os
from datetime import datetime
from config import DATABASE_PATH, DATA_FOLDER

class Database:
    """Database operations for InfraScope"""
    
    def __init__(self):
        self.db_path = DATABASE_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database with tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_path TEXT,
                cnn_defect TEXT,
                cnn_confidence REAL,
                cnn_severity TEXT,
                svm_defect TEXT,
                svm_confidence REAL,
                svm_severity TEXT,
                knn_defect TEXT,
                knn_confidence REAL,
                knn_severity TEXT,
                best_model TEXT,
                best_confidence REAL,
                prediction_time_ms REAL,
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                is_resolved BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id)
            )
        ''')
        
        # Create users table (for future authentication)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT,
                role TEXT DEFAULT 'viewer',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                avg_response_time_ms REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✓ Database initialized")
    
    def insert_prediction(self, data):
        """Store prediction in database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (
                    timestamp, filename, image_path,
                    cnn_defect, cnn_confidence, cnn_severity,
                    svm_defect, svm_confidence, svm_severity,
                    knn_defect, knn_confidence, knn_severity,
                    best_model, best_confidence,
                    prediction_time_ms, file_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('timestamp'),
                data.get('filename'),
                data.get('image_path'),
                data.get('cnn_defect'),
                data.get('cnn_confidence'),
                data.get('cnn_severity'),
                data.get('svm_defect'),
                data.get('svm_confidence'),
                data.get('svm_severity'),
                data.get('knn_defect'),
                data.get('knn_confidence'),
                data.get('knn_severity'),
                data.get('best_model'),
                data.get('best_confidence'),
                data.get('prediction_time_ms'),
                data.get('file_size')
            ))
            
            conn.commit()
            prediction_id = cursor.lastrowid
            conn.close()
            
            return prediction_id
        except Exception as e:
            print(f"Error inserting prediction: {e}")
            return None
    
    def get_predictions(self, limit=100, offset=0):
        """Get prediction history"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM predictions
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            rows = cursor.fetchall()
            predictions = [dict(row) for row in rows]
            conn.close()
            
            return predictions
        except Exception as e:
            print(f"Error retrieving predictions: {e}")
            return []
    
    def get_prediction_by_id(self, prediction_id):
        """Get specific prediction"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,))
            row = cursor.fetchone()
            conn.close()
            
            return dict(row) if row else None
        except Exception as e:
            print(f"Error retrieving prediction: {e}")
            return None
    
    def get_statistics(self):
        """Get database statistics"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute('SELECT COUNT(*) as count FROM predictions')
            total = cursor.fetchone()['count']
            
            # Average confidences
            cursor.execute('''
                SELECT 
                    AVG(cnn_confidence) as cnn_avg,
                    AVG(svm_confidence) as svm_avg,
                    AVG(knn_confidence) as knn_avg
                FROM predictions
            ''')
            
            stats_row = cursor.fetchone()
            conn.close()
            
            return {
                'total_predictions': total,
                'avg_confidence': {
                    'cnn': stats_row['cnn_avg'] or 0,
                    'svm': stats_row['svm_avg'] or 0,
                    'knn': stats_row['knn_avg'] or 0
                }
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def insert_alert(self, prediction_id, alert_type, severity, message):
        """Create alert for critical predictions"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (prediction_id, alert_type, severity, message)
                VALUES (?, ?, ?, ?)
            ''', (prediction_id, alert_type, severity, message))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error inserting alert: {e}")
            return False
    
    def get_active_alerts(self):
        """Get all unresolved alerts"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM alerts
                WHERE is_resolved = 0
                ORDER BY created_at DESC
            ''')
            
            rows = cursor.fetchall()
            alerts = [dict(row) for row in rows]
            conn.close()
            
            return alerts
        except Exception as e:
            print(f"Error retrieving alerts: {e}")
            return []
    
    def clear_old_predictions(self, days=30):
        """Clean up old predictions (archival)"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM predictions
                WHERE created_at < datetime('now', '-' || ? || ' days')
            ''', (days,))
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            return deleted
        except Exception as e:
            print(f"Error clearing old predictions: {e}")
            return 0
    
    def export_predictions_json(self, filename='predictions_export.json'):
        """Export predictions to JSON"""
        try:
            predictions = self.get_predictions(limit=10000)
            
            export_path = os.path.join(DATA_FOLDER, filename)
            with open(export_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            return export_path
        except Exception as e:
            print(f"Error exporting predictions: {e}")
            return None


# Global database instance
db = Database()
