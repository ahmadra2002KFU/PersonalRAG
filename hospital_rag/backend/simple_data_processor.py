"""
Simple Hospital Data Processor (without pandas dependency)
Processes CSV hospital data for RAG system ingestion
"""

import csv
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SimpleHospitalDataProcessor:
    """Process hospital CSV data for RAG ingestion without pandas"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.processed_documents = []
        
    def load_csv_data(self, filename: str) -> List[Dict[str, Any]]:
        """Load CSV file with error handling"""
        try:
            file_path = self.data_dir / filename
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                data = list(reader)
            logger.info(f"Loaded {len(data)} records from {filename}")
            return data
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []
    
    def process_patients_data(self) -> List[Dict[str, Any]]:
        """Process patient data into document format"""
        data = self.load_csv_data("hospital_patients.csv")
        if not data:
            return []
        
        documents = []
        for patient in data:
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
        data = self.load_csv_data("hospital_equipment.csv")
        if not data:
            return []
        
        documents = []
        for equipment in data:
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
                    "cost": int(equipment.get('Cost', 0)) if equipment.get('Cost', '').replace('.', '').isdigit() else 0,
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
        data = self.load_csv_data("hospital_operational_metrics.csv")
        if not data:
            return []
        
        documents = []
        for metric in data:
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
    
    def process_all_data(self) -> List[Dict[str, Any]]:
        """Process all hospital data files"""
        all_documents = []
        
        print("Processing hospital data files...")
        
        # Process each data type
        all_documents.extend(self.process_patients_data())
        all_documents.extend(self.process_equipment_data())
        all_documents.extend(self.process_metrics_data())
        
        print(f"Processed {len(all_documents)} total documents")
        
        return all_documents
    
    # Helper methods
    def _format_patient_content(self, patient: Dict[str, Any]) -> str:
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
    
    def _format_equipment_content(self, equipment: Dict[str, Any]) -> str:
        """Format equipment data into readable content"""
        content = f"Equipment {equipment['Equipment_ID']}: {equipment['Equipment_Name']}\n"
        content += f"Model: {equipment['Model']} by {equipment['Manufacturer']}\n"
        content += f"Department: {equipment['Department']}\n"
        content += f"Location: {equipment.get('Location', 'Unknown')}\n"
        content += f"Status: {equipment['Status']}\n"
        
        try:
            cost = int(float(equipment.get('Cost', 0)))
            content += f"Cost: ${cost:,}\n"
        except:
            content += f"Cost: {equipment.get('Cost', 'Unknown')}\n"
            
        content += f"Last Maintenance: {equipment.get('Last_Maintenance', 'Unknown')}\n"  
        content += f"Next Maintenance: {equipment.get('Next_Maintenance', 'Unknown')}\n"
        
        if equipment.get('Specifications'):
            content += f"Specifications: {equipment['Specifications']}\n"
        
        return content
    
    def _format_metric_content(self, metric: Dict[str, Any]) -> str:
        """Format metric data into readable content"""
        content = f"Metric: {metric['Metric_Name']}\n"
        content += f"Category: {metric['Metric_Category']}\n"
        content += f"Department: {metric['Department']}\n"
        content += f"Current Value: {metric['Current_Value']} {metric['Unit']}\n"
        content += f"Target Value: {metric['Target_Value']} {metric['Unit']}\n"
        content += f"Trend: {metric.get('Trend_Direction', 'Stable')}\n"
        
        if metric.get('Notes'):
            content += f"Notes: {metric['Notes']}\n"
        
        return content
    
    def _is_saudi_patient(self, patient: Dict[str, Any]) -> bool:
        """Determine if patient is Saudi based on name or phone"""
        saudi_indicators = ['Al-', '+966', 'Abdullah', 'Mohammed', 'Ahmed', 'Fatima', 'Aisha', 'Maryam']
        patient_text = f"{patient.get('First_Name', '')} {patient.get('Last_Name', '')} {patient.get('Phone', '')}"
        return any(indicator in patient_text for indicator in saudi_indicators)
    
    def _infer_department(self, patient: Dict[str, Any]) -> str:
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
        if not dob_str:
            return 'Unknown'
        
        try:
            # Simple age calculation
            year = int(dob_str.split('-')[0])
            current_year = datetime.now().year
            age = current_year - year
            
            if age < 18:
                return 'Pediatric'
            elif age < 65:
                return 'Adult'
            else:
                return 'Geriatric'
        except:
            return 'Unknown'
    
    def _get_equipment_criticality(self, equipment: Dict[str, Any]) -> str:
        """Determine equipment criticality level"""
        critical_equipment = ['ventilator', 'defibrillator', 'monitor', 'anesthesia']
        equipment_name = equipment.get('Equipment_Name', '').lower()
        
        if any(critical in equipment_name for critical in critical_equipment):
            return 'Critical'
        
        try:
            cost = float(equipment.get('Cost', 0))
            if cost > 100000:
                return 'High'
        except:
            pass
            
        return 'Medium'
    
    def _get_metric_priority(self, metric: Dict[str, Any]) -> str:
        """Determine metric priority based on category"""
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
        
        print(f"Saved {len(documents)} processed documents to {output_path}")


if __name__ == "__main__":
    # Test the processor
    processor = SimpleHospitalDataProcessor(Path("/mnt/z/Code/AI SIMA-RAG/0.1Ver"))
    documents = processor.process_all_data()
    processor.save_processed_data(documents)
    print(f"Processed {len(documents)} documents successfully!")