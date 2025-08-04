"""
Hospital Data Processing Pipeline
Processes CSV hospital data for RAG system ingestion
"""

import pandas as pd
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class HospitalDataProcessor:
    """Process hospital CSV data for RAG ingestion"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.processed_documents = []
        
    def load_csv_data(self, filename: str) -> pd.DataFrame:
        """Load CSV file with error handling"""
        try:
            file_path = self.data_dir / filename
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return pd.DataFrame()
    
    def process_patients_data(self) -> List[Dict[str, Any]]:
        """Process patient data into document format"""
        df = self.load_csv_data("hospital_patients.csv")
        if df.empty:
            return []
        
        documents = []
        for _, patient in df.iterrows():
            # Create main patient document
            doc = {
                "id": f"patient_{patient['Patient_ID']}",
                "type": "patient_record",
                "title": f"Patient Record: {patient['First_Name']} {patient['Last_Name']}",
                "content": self._format_patient_content(patient),
                "metadata": {
                    "patient_id": patient['Patient_ID'],
                    "department": self._infer_department(patient),
                    "status": patient.get('Current_Status', 'Unknown'),
                    "nationality": "Saudi" if self._is_saudi_patient(patient) else "Other",
                    "insurance": patient.get('Insurance_Type', 'Unknown'),
                    "blood_type": patient.get('Blood_Type', 'Unknown'),
                    "age_group": self._get_age_group(patient.get('Date_of_Birth')),
                    "has_allergies": patient.get('Allergies') != 'None',
                    "source": "hospital_patients.csv",
                    "last_updated": datetime.now().isoformat()
                }
            }
            documents.append(doc)
            
            # Create additional searchable documents for specific queries
            if patient.get('Current_Status') == 'Inpatient':
                inpatient_doc = {
                    "id": f"inpatient_{patient['Patient_ID']}",
                    "type": "inpatient_status",
                    "title": f"Current Inpatient: {patient['First_Name']} {patient['Last_Name']}",
                    "content": f"Currently admitted patient {patient['Patient_ID']} - {patient['First_Name']} {patient['Last_Name']}. Status: {patient['Current_Status']}. Admission Date: {patient.get('Admission_Date', 'Not specified')}.",
                    "metadata": {**doc["metadata"], "query_type": "current_patients"}
                }
                documents.append(inpatient_doc)
        
        return documents
    
    def process_equipment_data(self) -> List[Dict[str, Any]]:
        """Process equipment data into document format"""
        df = self.load_csv_data("hospital_equipment.csv")
        if df.empty:
            return []
        
        documents = []
        for _, equipment in df.iterrows():
            doc = {
                "id": f"equipment_{equipment['Equipment_ID']}",
                "type": "equipment_record",
                "title": f"{equipment['Equipment_Name']} - {equipment['Model']}",
                "content": self._format_equipment_content(equipment),
                "metadata": {
                    "equipment_id": equipment['Equipment_ID'],
                    "department": equipment['Department'],
                    "status": equipment['Status'],
                    "manufacturer": equipment['Manufacturer'],
                    "cost": equipment.get('Cost', 0),
                    "criticality": self._get_equipment_criticality(equipment),
                    "maintenance_due": equipment.get('Next_Maintenance'),
                    "location": equipment.get('Location', 'Unknown'),
                    "source": "hospital_equipment.csv",
                    "last_updated": datetime.now().isoformat()
                }
            }
            documents.append(doc)
            
            # Create status-specific documents
            if equipment['Status'] == 'Maintenance Required':
                maintenance_doc = {
                    "id": f"maintenance_{equipment['Equipment_ID']}",
                    "type": "maintenance_alert",
                    "title": f"Maintenance Required: {equipment['Equipment_Name']}",
                    "content": f"Equipment {equipment['Equipment_ID']} ({equipment['Equipment_Name']}) in {equipment['Department']} requires maintenance. Status: {equipment['Status']}. Last maintenance: {equipment.get('Last_Maintenance', 'Unknown')}.",
                    "metadata": {**doc["metadata"], "query_type": "maintenance_alerts"}
                }
                documents.append(maintenance_doc)
        
        return documents
    
    def process_metrics_data(self) -> List[Dict[str, Any]]:
        """Process operational metrics into document format"""
        df = self.load_csv_data("hospital_operational_metrics.csv")
        if df.empty:
            return []
        
        documents = []
        for _, metric in df.iterrows():
            doc = {
                "id": f"metric_{metric['Metric_Category']}_{metric['Metric_Name']}".replace(' ', '_').lower(),
                "type": "operational_metric",
                "title": f"{metric['Metric_Category']}: {metric['Metric_Name']}",
                "content": self._format_metric_content(metric),
                "metadata": {
                    "category": metric['Metric_Category'],
                    "metric_name": metric['Metric_Name'],
                    "current_value": metric['Current_Value'],
                    "target_value": metric['Target_Value'],
                    "unit": metric['Unit'],
                    "department": metric['Department'],
                    "trend": metric.get('Trend_Direction', 'Stable'),
                    "priority": self._get_metric_priority(metric),
                    "source": "hospital_operational_metrics.csv",
                    "last_updated": datetime.now().isoformat()
                }
            }
            documents.append(doc)
        
        return documents
    
    def process_appointments_data(self) -> List[Dict[str, Any]]:
        """Process appointment data into document format"""
        df = self.load_csv_data("hospital_appointments.csv")
        if df.empty:
            return []
        
        documents = []
        for _, appointment in df.iterrows():
            doc = {
                "id": f"appointment_{appointment['Appointment_ID']}",
                "type": "appointment_record",
                "title": f"Appointment {appointment['Appointment_ID']} - {appointment['Department']}",
                "content": self._format_appointment_content(appointment),
                "metadata": {
                    "appointment_id": appointment['Appointment_ID'],
                    "patient_id": appointment['Patient_ID'],
                    "doctor_id": appointment['Doctor_ID'],
                    "department": appointment['Department'],
                    "status": appointment['Status'],
                    "appointment_type": appointment['Appointment_Type'],
                    "date": appointment['Date'],
                    "time": appointment['Time'],
                    "source": "hospital_appointments.csv",
                    "last_updated": datetime.now().isoformat()
                }
            }
            documents.append(doc)
        
        return documents
    
    def process_medical_records_data(self) -> List[Dict[str, Any]]:
        """Process medical records into document format"""
        df = self.load_csv_data("hospital_medical_records.csv")
        if df.empty:
            return []
        
        documents = []
        for _, record in df.iterrows():
            doc = {
                "id": f"medical_record_{record['Record_ID']}",
                "type": "medical_record",
                "title": f"Medical Record {record['Record_ID']} - {record['Diagnosis']}",
                "content": self._format_medical_record_content(record),
                "metadata": {
                    "record_id": record['Record_ID'],
                    "patient_id": record['Patient_ID'],
                    "doctor_id": record['Doctor_ID'],
                    "diagnosis": record['Diagnosis'],
                    "treatment": record['Treatment'],
                    "date": record['Date'],
                    "follow_up_required": bool(record.get('Follow_up_Date')),
                    "source": "hospital_medical_records.csv",
                    "last_updated": datetime.now().isoformat()
                }
            }
            documents.append(doc)
        
        return documents
    
    def process_all_data(self) -> List[Dict[str, Any]]:
        """Process all hospital data files"""
        all_documents = []
        
        logger.info("Processing hospital data files...")
        
        # Process each data type
        all_documents.extend(self.process_patients_data())
        all_documents.extend(self.process_equipment_data())
        all_documents.extend(self.process_metrics_data())
        all_documents.extend(self.process_appointments_data())
        all_documents.extend(self.process_medical_records_data())
        
        logger.info(f"Processed {len(all_documents)} total documents")
        
        return all_documents
    
    # Helper methods
    def _format_patient_content(self, patient: pd.Series) -> str:
        """Format patient data into readable content"""
        content = f"Patient {patient['Patient_ID']}: {patient['First_Name']} {patient['Last_Name']}\n"
        content += f"Date of Birth: {patient['Date_of_Birth']}\n"
        content += f"Gender: {patient['Gender']}\n"
        content += f"Blood Type: {patient.get('Blood_Type', 'Unknown')}\n"
        content += f"Insurance: {patient.get('Insurance_Type', 'Unknown')}\n"
        content += f"Allergies: {patient.get('Allergies', 'Unknown')}\n"
        content += f"Current Status: {patient.get('Current_Status', 'Unknown')}\n"
        
        if patient.get('Admission_Date'):
            content += f"Admission Date: {patient['Admission_Date']}\n"
        if patient.get('Discharge_Date'):
            content += f"Discharge Date: {patient['Discharge_Date']}\n"
        
        return content
    
    def _format_equipment_content(self, equipment: pd.Series) -> str:
        """Format equipment data into readable content"""
        content = f"Equipment {equipment['Equipment_ID']}: {equipment['Equipment_Name']}\n"
        content += f"Model: {equipment['Model']} by {equipment['Manufacturer']}\n"
        content += f"Department: {equipment['Department']}\n"
        content += f"Location: {equipment.get('Location', 'Unknown')}\n"
        content += f"Status: {equipment['Status']}\n"
        content += f"Cost: ${equipment.get('Cost', 0):,}\n"
        content += f"Last Maintenance: {equipment.get('Last_Maintenance', 'Unknown')}\n"
        content += f"Next Maintenance: {equipment.get('Next_Maintenance', 'Unknown')}\n"
        
        if 'Specifications' in equipment and pd.notna(equipment['Specifications']):
            content += f"Specifications: {equipment['Specifications']}\n"
        
        return content
    
    def _format_metric_content(self, metric: pd.Series) -> str:
        """Format metric data into readable content"""
        content = f"Metric: {metric['Metric_Name']}\n"
        content += f"Category: {metric['Metric_Category']}\n"
        content += f"Department: {metric['Department']}\n"
        content += f"Current Value: {metric['Current_Value']} {metric['Unit']}\n"
        content += f"Target Value: {metric['Target_Value']} {metric['Unit']}\n"
        content += f"Trend: {metric.get('Trend_Direction', 'Stable')}\n"
        
        if 'Notes' in metric and pd.notna(metric['Notes']):
            content += f"Notes: {metric['Notes']}\n"
        
        return content
    
    def _format_appointment_content(self, appointment: pd.Series) -> str:
        """Format appointment data into readable content"""
        content = f"Appointment {appointment['Appointment_ID']}\n"
        content += f"Patient: {appointment['Patient_ID']}\n"
        content += f"Doctor: {appointment['Doctor_ID']}\n"
        content += f"Department: {appointment['Department']}\n"
        content += f"Date: {appointment['Date']} at {appointment['Time']}\n"
        content += f"Type: {appointment['Appointment_Type']}\n"
        content += f"Status: {appointment['Status']}\n"
        content += f"Duration: {appointment.get('Duration_Minutes', 'Unknown')} minutes\n"
        
        if 'Reason' in appointment and pd.notna(appointment['Reason']):
            content += f"Reason: {appointment['Reason']}\n"
        
        return content
    
    def _format_medical_record_content(self, record: pd.Series) -> str:
        """Format medical record data into readable content"""
        content = f"Medical Record {record['Record_ID']}\n"
        content += f"Patient: {record['Patient_ID']}\n"
        content += f"Doctor: {record['Doctor_ID']}\n"
        content += f"Date: {record['Date']}\n"
        content += f"Diagnosis: {record['Diagnosis']}\n"
        content += f"Treatment: {record['Treatment']}\n"
        content += f"Medications: {record.get('Medications', 'None specified')}\n"
        content += f"Vital Signs: {record.get('Vital_Signs', 'Not recorded')}\n"
        
        if 'Lab_Results' in record and pd.notna(record['Lab_Results']):
            content += f"Lab Results: {record['Lab_Results']}\n"
        if 'Notes' in record and pd.notna(record['Notes']):
            content += f"Clinical Notes: {record['Notes']}\n"
        
        return content
    
    def _is_saudi_patient(self, patient: pd.Series) -> bool:
        """Determine if patient is Saudi based on name or phone"""
        saudi_indicators = ['Al-', '+966', 'Abdullah', 'Mohammed', 'Ahmed', 'Fatima', 'Aisha', 'Maryam']
        patient_text = f"{patient.get('First_Name', '')} {patient.get('Last_Name', '')} {patient.get('Phone', '')}"
        return any(indicator in patient_text for indicator in saudi_indicators)
    
    def _infer_department(self, patient: pd.Series) -> str:
        """Infer likely department based on patient status"""
        status = patient.get('Current_Status', '').lower()
        if 'emergency' in status:
            return 'Emergency'
        elif 'inpatient' in status:
            return 'General Medicine'
        else:
            return 'Outpatient'
    
    def _get_age_group(self, dob_str: Optional[str]) -> str:
        """Calculate age group from date of birth"""
        if not dob_str or pd.isna(dob_str):
            return 'Unknown'
        
        try:
            dob = pd.to_datetime(dob_str)
            age = (datetime.now() - dob).days // 365
            
            if age < 18:
                return 'Pediatric'
            elif age < 65:
                return 'Adult'
            else:
                return 'Geriatric'
        except:
            return 'Unknown'
    
    def _get_equipment_criticality(self, equipment: pd.Series) -> str:
        """Determine equipment criticality level"""
        critical_equipment = ['ventilator', 'defibrillator', 'monitor', 'anesthesia']
        equipment_name = equipment.get('Equipment_Name', '').lower()
        
        if any(critical in equipment_name for critical in critical_equipment):
            return 'Critical'
        elif equipment.get('Cost', 0) > 100000:
            return 'High'
        else:
            return 'Medium'
    
    def _get_metric_priority(self, metric: pd.Series) -> str:
        """Determine metric priority based on category and values"""
        category = metric.get('Metric_Category', '').lower()
        
        if 'safety' in category or 'quality' in category:
            return 'High'
        elif 'financial' in category or 'staff' in category:
            return 'Medium'
        else:
            return 'Low'
    
    def save_processed_data(self, documents: List[Dict[str, Any]], output_file: str = "processed_hospital_data.json"):
        """Save processed documents to JSON file"""
        output_path = self.data_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(documents)} processed documents to {output_path}")


if __name__ == "__main__":
    # Test the processor
    processor = HospitalDataProcessor(Path("/mnt/z/Code/AI SIMA-RAG/0.1Ver"))
    documents = processor.process_all_data()
    processor.save_processed_data(documents)
    print(f"Processed {len(documents)} documents successfully!")