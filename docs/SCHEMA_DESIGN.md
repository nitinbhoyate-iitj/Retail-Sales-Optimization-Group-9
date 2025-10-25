# MongoDB Schema Design

## Overview

This document describes the MongoDB collections schema for the Retail Sales Optimization project. The database uses a flexible document-based structure optimized for analytical queries and reporting.

## Database: `retail_sales_db`

---

## Collections

### 1. `raw_sales`

**Purpose:** Store raw, unprocessed data from source files.

**Schema:**
```javascript
{
  invoice_no: String,              // Invoice number (e.g., "I138884")
  customer_id: String,             // Customer identifier
  gender: String,                  // Customer gender
  age: Number,                     // Customer age
  category: String,                // Product category
  quantity: Number,                // Quantity purchased
  price: Number,                   // Price per unit
  payment_method: String,          // Payment method used
  invoice_date: Date,              // Transaction date
  shopping_mall: String            // Shopping mall location
}
```

**Indexes:**
- None (raw data collection)

**Sample Document:**
```json
{
  "invoice_no": "I138884",
  "customer_id": "C241288",
  "gender": "Female",
  "age": 28,
  "category": "Clothing",
  "quantity": 5,
  "price": 1500.40,
  "payment_method": "Credit Card",
  "invoice_date": ISODate("2022-03-08T00:00:00Z"),
  "shopping_mall": "Kanyon"
}
```

---

### 2. `cleaned_sales`

**Purpose:** Store cleaned and standardized data ready for analysis.

**Schema:**
```javascript
{
  invoice_no: String,              // Standardized invoice number
  customer_id: String,             // Standardized customer ID
  gender: String,                  // Standardized gender (lowercase)
  age: Number (Int32),             // Validated age
  category: String,                // Standardized category (lowercase)
  quantity: Number (Int32),        // Validated quantity
  price: Number (Double),          // Validated price
  total_amount: Number (Double),   // Calculated: price * quantity
  payment_method: String,          // Standardized payment method
  invoice_date: Date,              // Standardized date
  shopping_mall: String,           // Standardized mall name
  year: Number (Int32),            // Extracted year
  month: Number (Int32),           // Extracted month (1-12)
  day: Number (Int32),             // Extracted day
  day_of_week: Number (Int32),     // Day of week (0-6, Monday=0)
  quarter: Number (Int32)          // Quarter (1-4)
}
```

**Indexes:**
```javascript
// Primary query indexes
db.cleaned_sales.createIndex({ "invoice_date": 1 })
db.cleaned_sales.createIndex({ "customer_id": 1 })
db.cleaned_sales.createIndex({ "category": 1 })
db.cleaned_sales.createIndex({ "shopping_mall": 1 })

// Compound indexes for common queries
db.cleaned_sales.createIndex({ "category": 1, "invoice_date": 1 })
db.cleaned_sales.createIndex({ "customer_id": 1, "invoice_date": 1 })
db.cleaned_sales.createIndex({ "year": 1, "month": 1 })
```

**Sample Document:**
```json
{
  "invoice_no": "I138884",
  "customer_id": "C241288",
  "gender": "female",
  "age": 28,
  "category": "clothing",
  "quantity": 5,
  "price": 1500.40,
  "total_amount": 7502.00,
  "payment_method": "credit card",
  "invoice_date": ISODate("2022-03-08T00:00:00Z"),
  "shopping_mall": "kanyon",
  "year": 2022,
  "month": 3,
  "day": 8,
  "day_of_week": 1,
  "quarter": 1
}
```

---

### 3. `transformed_sales`

**Purpose:** Store feature-engineered data with additional analytical features.

**Schema:**
```javascript
{
  // Base fields from cleaned_sales
  invoice_no: String,
  customer_id: String,
  gender: String,
  age: Number (Int32),
  category: String,
  quantity: Number (Int32),
  price: Number (Double),
  total_amount: Number (Double),
  payment_method: String,
  invoice_date: Date,
  shopping_mall: String,
  
  // Time features
  year: Number (Int32),
  month: Number (Int32),
  day: Number (Int32),
  day_of_week: Number (Int32),
  day_of_year: Number (Int32),
  week_of_year: Number (Int32),
  quarter: Number (Int32),
  is_weekend: Number (Int32),        // 0 or 1
  is_month_start: Number (Int32),    // 0 or 1
  is_month_end: Number (Int32),      // 0 or 1
  season: Number (Int32),            // 1-4
  
  // Customer features (RFM)
  recency: Number (Int32),           // Days since last purchase
  frequency: Number (Int32),         // Number of purchases
  monetary: Number (Double),         // Total spending
  customer_avg_amount: Number,       // Customer's avg transaction
  customer_std_amount: Number,       // Std dev of transactions
  customer_min_amount: Number,       // Min transaction
  customer_max_amount: Number,       // Max transaction
  customer_ltv: Number (Double),     // Customer lifetime value
  
  // Category features
  category_avg_amount: Number,       // Category average
  category_total_amount: Number,     // Category total
  category_transaction_count: Number,// Number of transactions
  category_rank: Number,             // Category rank by sales
  price_vs_category_avg: Number,     // Price relative to category avg
  
  // Lag features
  total_amount_lag_1: Number,        // Previous day amount
  total_amount_lag_7: Number,        // 7 days ago amount
  total_amount_lag_14: Number,       // 14 days ago amount
  
  // Rolling features
  total_amount_rolling_mean_7: Number,   // 7-day rolling average
  total_amount_rolling_std_7: Number,    // 7-day rolling std
  total_amount_rolling_mean_14: Number,  // 14-day rolling average
  total_amount_rolling_std_14: Number,   // 14-day rolling std
  total_amount_rolling_mean_30: Number,  // 30-day rolling average
  total_amount_rolling_std_30: Number,   // 30-day rolling std
  
  // Encoded features
  gender_encoded: Number (Int32),
  category_encoded: Number (Int32),
  payment_method_encoded: Number (Int32),
  shopping_mall_encoded: Number (Int32)
}
```

**Indexes:**
```javascript
// Core indexes
db.transformed_sales.createIndex({ "invoice_date": 1 })
db.transformed_sales.createIndex({ "customer_id": 1 })
db.transformed_sales.createIndex({ "category": 1 })

// Feature-specific indexes
db.transformed_sales.createIndex({ "recency": 1, "frequency": 1, "monetary": 1 })
db.transformed_sales.createIndex({ "customer_ltv": -1 })
db.transformed_sales.createIndex({ "category_rank": 1 })
```

---

### 4. `aggregated_sales`

**Purpose:** Pre-computed aggregations for dashboard performance.

**Schema:**
```javascript
{
  aggregation_type: String,        // Type of aggregation
  period: String,                  // Time period (if applicable)
  dimension: String,               // Dimension (category, customer, etc.)
  dimension_value: String,         // Value of dimension
  metrics: {
    total_sales: Number (Double),
    total_quantity: Number (Int32),
    transaction_count: Number (Int32),
    unique_customers: Number (Int32),
    avg_transaction: Number (Double),
    min_transaction: Number (Double),
    max_transaction: Number (Double)
  },
  created_at: Date                 // Timestamp
}
```

**Indexes:**
```javascript
db.aggregated_sales.createIndex({ "aggregation_type": 1, "dimension": 1 })
db.aggregated_sales.createIndex({ "period": 1 })
```

---

### 5. `agg_daily_sales`

**Purpose:** Daily aggregated sales metrics.

**Schema:**
```javascript
{
  date: Date,                      // Date
  total_sales: Number (Double),    // Total sales
  avg_transaction: Number,         // Average transaction
  transaction_count: Number,       // Number of transactions
  total_quantity: Number,          // Total items sold
  avg_quantity: Number,            // Average items per transaction
  unique_customers: Number         // Number of unique customers
}
```

**Indexes:**
```javascript
db.agg_daily_sales.createIndex({ "date": 1 })
```

---

### 6. `agg_category_sales`

**Purpose:** Category-level aggregated metrics.

**Schema:**
```javascript
{
  category: String,                // Category name
  total_sales: Number (Double),    // Total sales
  avg_transaction: Number,         // Average transaction
  transaction_count: Number,       // Number of transactions
  total_quantity: Number,          // Total items sold
  avg_quantity: Number,            // Average items per transaction
  unique_customers: Number         // Number of unique customers
}
```

**Indexes:**
```javascript
db.agg_category_sales.createIndex({ "category": 1 })
db.agg_category_sales.createIndex({ "total_sales": -1 })
```

---

### 7. `agg_customer_summary`

**Purpose:** Customer-level summary statistics.

**Schema:**
```javascript
{
  customer_id: String,             // Customer ID
  total_spent: Number (Double),    // Total amount spent
  avg_transaction: Number,         // Average transaction value
  transaction_count: Number,       // Number of purchases
  first_purchase: Date,            // Date of first purchase
  last_purchase: Date,             // Date of last purchase
  days_active: Number,             // Days between first and last
  recency: Number,                 // Days since last purchase
  avg_items_per_transaction: Number,
  preferred_category: String,      // Most purchased category
  preferred_payment_method: String,
  preferred_mall: String
}
```

**Indexes:**
```javascript
db.agg_customer_summary.createIndex({ "customer_id": 1 })
db.agg_customer_summary.createIndex({ "total_spent": -1 })
db.agg_customer_summary.createIndex({ "recency": 1, "total_spent": -1 })
```

---

## Data Types Reference

| Field Type | MongoDB Type | Description |
|-----------|--------------|-------------|
| String | String | Text data |
| Number (Int32) | 32-bit integer | Whole numbers |
| Number (Double) | 64-bit float | Decimal numbers |
| Date | ISODate | Date/time values |

---

## Query Optimization Guidelines

### 1. Use Covered Queries
Ensure indexes cover all query fields to avoid document scans.

### 2. Projection
Always use projection to return only needed fields:
```javascript
db.cleaned_sales.find(
  { category: "clothing" },
  { invoice_no: 1, total_amount: 1, invoice_date: 1 }
)
```

### 3. Aggregation Pipeline
Use aggregation pipelines for complex queries:
```javascript
db.cleaned_sales.aggregate([
  { $match: { year: 2022 } },
  { $group: {
      _id: "$category",
      total_sales: { $sum: "$total_amount" }
  }},
  { $sort: { total_sales: -1 } }
])
```

### 4. Index Usage
Monitor index usage with `explain()`:
```javascript
db.cleaned_sales.find({ customer_id: "C123" }).explain("executionStats")
```

---

## Maintenance

### Regular Tasks

1. **Rebuild Indexes** (monthly)
```javascript
db.cleaned_sales.reIndex()
```

2. **Check Index Usage**
```javascript
db.cleaned_sales.aggregate([{ $indexStats: {} }])
```

3. **Compact Collections** (as needed)
```javascript
db.runCommand({ compact: 'cleaned_sales' })
```

4. **Monitor Collection Stats**
```javascript
db.cleaned_sales.stats()
```

---

## Backup Strategy

1. **Daily backups** of raw_sales collection
2. **Weekly backups** of all collections
3. **Retention**: 30 days for daily, 90 days for weekly
4. Use MongoDB backup tools or AWS DocumentDB snapshots

---

## Data Retention Policy

- **raw_sales**: Retain indefinitely (archive after 2 years)
- **cleaned_sales**: Retain for 2 years
- **transformed_sales**: Retain for 1 year
- **aggregated collections**: Retain for 6 months
- **logs**: Retain for 3 months

---

Last Updated: October 2025

