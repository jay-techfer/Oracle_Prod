CREATE TABLE HumanResources_Department (
    DepartmentID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    Name VARCHAR2(50) NOT NULL,
    GroupName VARCHAR2(50) NOT NULL,
    ModifiedDate DATE NOT NULL
);

CREATE TABLE HumanResources_Employee (
    BusinessEntityID NUMBER PRIMARY KEY,
    NationalIDNumber VARCHAR2(15) NOT NULL,
    LoginID VARCHAR2(256) NOT NULL,
    OrganizationNode VARCHAR2(100),
    OrganizationLevel NUMBER,
    JobTitle VARCHAR2(100) NOT NULL,
    BirthDate DATE NOT NULL,
    MaritalStatus CHAR(1) NOT NULL,
    Gender CHAR(1) NOT NULL,
    HireDate DATE NOT NULL,
    SalariedFlag NUMBER(1) NOT NULL,
    VacationHours NUMBER NOT NULL,
    SickLeaveHours NUMBER NOT NULL,
    CurrentFlag NUMBER(1) NOT NULL,
    rowguid RAW(16) NOT NULL,
    ModifiedDate DATE NOT NULL
);

CREATE TABLE HumanResources_EmployeeDepartmentHistory (
    BusinessEntityID NUMBER NOT NULL,
    DepartmentID NUMBER NOT NULL,
    ShiftID NUMBER NOT NULL,
    StartDate DATE NOT NULL,
    EndDate DATE,
    ModifiedDate DATE NOT NULL,
    PRIMARY KEY (BusinessEntityID, DepartmentID, ShiftID, StartDate)
);

CREATE TABLE HumanResources_EmployeePayHistory (
    BusinessEntityID NUMBER NOT NULL,
    RateChangeDate DATE NOT NULL,
    Rate NUMBER(10,4) NOT NULL,
    PayFrequency NUMBER(1) NOT NULL,
    ModifiedDate DATE NOT NULL,
    PRIMARY KEY (BusinessEntityID, RateChangeDate)
);

CREATE TABLE Shift (
    ShiftID        NUMBER PRIMARY KEY,
    Name           VARCHAR2(50),
    StartTime      DATE,
    EndTime        DATE,
    ModifiedDate   DATE
);

CREATE TABLE Address (
    AddressID       NUMBER PRIMARY KEY,
    AddressLine1    VARCHAR2(100) NOT NULL,
    AddressLine2    VARCHAR2(100),
    City            VARCHAR2(50) NOT NULL,
    StateProvinceID NUMBER NOT NULL,
    PostalCode      VARCHAR2(20) NOT NULL,
    SpatialLocation RAW(100),
    rowguid         RAW(16) NOT NULL,
    ModifiedDate    DATE NOT NULL
);

CREATE TABLE AddressType (
    AddressTypeID NUMBER PRIMARY KEY,
    Name          VARCHAR2(50) NOT NULL,
    rowguid       RAW(16) NOT NULL,
    ModifiedDate  DATE NOT NULL
);

CREATE TABLE BusinessEntity (
    BusinessEntityID NUMBER PRIMARY KEY,
    rowguid          RAW(16) NOT NULL,
    ModifiedDate     DATE NOT NULL
);

CREATE TABLE BusinessEntityAddress (
    BusinessEntityID NUMBER NOT NULL,
    AddressID        NUMBER NOT NULL,
    AddressTypeID    NUMBER NOT NULL,
    rowguid          RAW(16) NOT NULL,
    ModifiedDate     DATE NOT NULL,
    PRIMARY KEY (BusinessEntityID, AddressID, AddressTypeID)
);
CREATE TABLE BusinessEntityContact (
    BusinessEntityID NUMBER NOT NULL,
    PersonID         NUMBER NOT NULL,
    ContactTypeID    NUMBER NOT NULL,
    rowguid          RAW(16) NOT NULL,
    ModifiedDate     DATE NOT NULL,
    PRIMARY KEY (BusinessEntityID, PersonID, ContactTypeID)
);

CREATE TABLE ContactType (
    ContactTypeID VARCHAR2(10) PRIMARY KEY,
    Name          VARCHAR2(100) NOT NULL,
    ModifiedDate  DATE NOT NULL
);
CREATE TABLE CountryRegion (
    CountryRegionCode VARCHAR2(5) PRIMARY KEY,
    Name              VARCHAR2(100) NOT NULL,
    ModifiedDate      DATE NOT NULL
);

CREATE TABLE EmailAddress (
    BusinessEntityID NUMBER NOT NULL,
    EmailAddressID   NUMBER NOT NULL,
    EmailAddress     VARCHAR2(100) NOT NULL,
    rowguid          RAW(16) NOT NULL,
    ModifiedDate     DATE NOT NULL,
    PRIMARY KEY (BusinessEntityID, EmailAddressID)
);

CREATE TABLE Password (
    BusinessEntityID NUMBER NOT NULL,
    PasswordHash     VARCHAR2(200) NOT NULL,
    PasswordSalt     VARCHAR2(50) NOT NULL,
    rowguid          RAW(16) NOT NULL,
    ModifiedDate     DATE NOT NULL,
    PRIMARY KEY (BusinessEntityID)
);

CREATE TABLE Person (
    BusinessEntityID      NUMBER PRIMARY KEY,
    PersonType            VARCHAR2(2) NOT NULL,
    NameStyle             NUMBER(1) NOT NULL,
    Title                 VARCHAR2(8),
    FirstName             VARCHAR2(50) NOT NULL,
    MiddleName            VARCHAR2(50),
    LastName              VARCHAR2(50) NOT NULL,
    Suffix                VARCHAR2(10),
    EmailPromotion        NUMBER(1) NOT NULL,
    AdditionalContactInfo CLOB,
    Demographics          CLOB,
    rowguid               RAW(16) NOT NULL,
    ModifiedDate          DATE NOT NULL
);
CREATE TABLE BusinessEntityPhone (
    BusinessEntityID   NUMBER NOT NULL,
    PhoneNumber        VARCHAR2(25) NOT NULL,
    PhoneNumberTypeID  NUMBER NOT NULL,
    ModifiedDate       DATE NOT NULL,
    PRIMARY KEY (BusinessEntityID, PhoneNumber, PhoneNumberTypeID)
);

CREATE TABLE PhoneNumberType (
    PhoneNumberTypeID NUMBER PRIMARY KEY,
    Name              VARCHAR2(50) NOT NULL,
    ModifiedDate      DATE NOT NULL
);

CREATE TABLE StateProvince (
    StateProvinceID          NUMBER PRIMARY KEY,
    StateProvinceCode        VARCHAR2(5) NOT NULL,
    CountryRegionCode        VARCHAR2(5) NOT NULL,
    IsOnlyStateProvinceFlag  NUMBER(1) NOT NULL,
    Name                     VARCHAR2(100) NOT NULL,
    TerritoryID              NUMBER NOT NULL,
    rowguid                  RAW(16) NOT NULL,
    ModifiedDate             DATE NOT NULL
);

CREATE TABLE BillOfMaterials (
    BillOfMaterialsID   NUMBER PRIMARY KEY,
    ProductAssemblyID   NUMBER,
    ComponentID         NUMBER NOT NULL,
    StartDate           DATE NOT NULL,
    EndDate             DATE,
    UnitMeasureCode     VARCHAR2(10) NOT NULL,
    BOMLevel            NUMBER(3) NOT NULL,
    PerAssemblyQty      NUMBER(10,4) NOT NULL,
    ModifiedDate        DATE NOT NULL
);

CREATE TABLE Culture (
    CultureID     VARCHAR2(10) PRIMARY KEY,
    Name          VARCHAR2(100) NOT NULL,
    ModifiedDate  DATE NOT NULL
);

CREATE TABLE ProductCategory (
    ProductCategoryID NUMBER PRIMARY KEY,
    "Name" VARCHAR2(100),
    rowguid RAW(16),
    ModifiedDate DATE
);

CREATE TABLE ProductDescription (
    ProductDescriptionID NUMBER PRIMARY KEY,
    Description          VARCHAR2(4000),  -- Big enough for long text
    rowguid              RAW(16) NOT NULL,
    ModifiedDate         DATE NOT NULL
);

CREATE TABLE ProductInventory (
    ProductID     NUMBER NOT NULL,
    LocationID    NUMBER NOT NULL,
    Shelf         VARCHAR2(10) NOT NULL,
    Bin           NUMBER NOT NULL,
    Quantity      NUMBER NOT NULL,
    rowguid       RAW(16) NOT NULL,
    ModifiedDate  DATE NOT NULL,
    CONSTRAINT PK_ProductInventory PRIMARY KEY (ProductID, LocationID, Shelf, Bin)
);



CREATE TABLE Product (
    ProductID NUMBER PRIMARY KEY,
    ProductName VARCHAR2(100),
    ProductNumber VARCHAR2(25),
    MakeFlag NUMBER(1),
    FinishedGoodsFlag NUMBER(1),
    Color VARCHAR2(50),
    SafetyStockLevel NUMBER,
    ReorderPoint NUMBER,
    StandardCost NUMBER(18,2),
    ListPrice NUMBER(18,2),
    SizeNO VARCHAR2(10),
    SizeUnitMeasureCode VARCHAR2(15),
    WeightUnitMeasureCode VARCHAR2(15),
    Weight NUMBER(18,2),
    DaysToManufacture NUMBER,
    ClassName VARCHAR2(10),
    StyleName VARCHAR2(10),
    ProductSubcategoryID NUMBER,
    ProductModelID NUMBER,
    SellStartDate  DATE,
    SellEndDate DATE,
    DiscontinuedDate DATE,
    rowguid RAW(16),
    ModifiedDate TIMESTAMP
);

CREATE TABLE Sales_CountryRegionCurrency (
    CountryRegionCode VARCHAR2(3) NOT NULL,
    CurrencyCode VARCHAR2(3) NOT NULL,
    ModifiedDate DATE NOT NULL,
    CONSTRAINT PK_CountryRegionCurrency PRIMARY KEY (CountryRegionCode, CurrencyCode)
);

CREATE TABLE Sales_CreditCard (
    CreditCardID NUMBER PRIMARY KEY,
    CardType VARCHAR2(50) NOT NULL,
    CardNumber VARCHAR2(25) NOT NULL,
    ExpMonth NUMBER(2) NOT NULL,
    ExpYear NUMBER(4) NOT NULL,
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Sales_Currency (
    CurrencyCode VARCHAR2(3) PRIMARY KEY,
    Name VARCHAR2(50) NOT NULL,
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Sales_CurrencyRate (
    CurrencyRateID NUMBER PRIMARY KEY,
    CurrencyRateDate DATE NOT NULL,
    FromCurrencyCode VARCHAR2(3) NOT NULL,
    ToCurrencyCode VARCHAR2(3) NOT NULL,
    AverageRate NUMBER(19,4) NOT NULL,
    EndOfDayRate NUMBER(19,4) NOT NULL,
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Sales_Customer (
    CustomerID NUMBER PRIMARY KEY,
    PersonID NUMBER,
    StoreID NUMBER,
    TerritoryID NUMBER,
    AccountNumber VARCHAR2(10) NOT NULL,
    rowguid RAW(16) NOT NULL,
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Sales_PersonCreditCard (
    BusinessEntityID NUMBER NOT NULL,
    CreditCardID NUMBER NOT NULL,
    ModifiedDate DATE NOT NULL,
    CONSTRAINT PK_PersonCreditCard PRIMARY KEY (BusinessEntityID, CreditCardID)
);

CREATE TABLE Sales_SalesOrderDetail (
    SalesOrderID NUMBER NOT NULL,
    SalesOrderDetailID NUMBER PRIMARY KEY,
    CarrierTrackingNumber VARCHAR2(25),
    OrderQty NUMBER(4) NOT NULL,
    ProductID NUMBER NOT NULL,
    SpecialOfferID NUMBER NOT NULL,
    UnitPrice NUMBER(19,4) NOT NULL,
    UnitPriceDiscount NUMBER(19,4) NOT NULL,
    LineTotal NUMBER(38,6) NOT NULL,
    rowguid RAW(16) NOT NULL,
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Sales_SalesOrderHeader (
    SalesOrderID NUMBER PRIMARY KEY,
    RevisionNumber NUMBER(3) NOT NULL,
    OrderDate DATE NOT NULL,
    DueDate DATE NOT NULL,
    ShipDate DATE,
    Status NUMBER(2) NOT NULL,
    OnlineOrderFlag CHAR(1) NOT NULL,
    SalesOrderNumber VARCHAR2(25 CHAR) NOT NULL,
    PurchaseOrderNumber VARCHAR2(25 CHAR),
    AccountNumber VARCHAR2(25 CHAR),
    CustomerID NUMBER NOT NULL,
    SalesPersonID NUMBER,
    TerritoryID NUMBER,
    BillToAddressID NUMBER NOT NULL,
    ShipToAddressID NUMBER NOT NULL,
    ShipMethodID NUMBER NOT NULL,
    CreditCardID NUMBER,
    CreditCardApprovalCode VARCHAR2(15 CHAR),
    CurrencyRateID NUMBER,
    SubTotal NUMBER(14,2) NOT NULL,
    TaxAmt NUMBER(12,2) NOT NULL,
    Freight NUMBER(12,2) NOT NULL,
    TotalDue NUMBER(14,2) NOT NULL,
    "Comment" VARCHAR2(128 CHAR),
    rowguid VARCHAR2(36 CHAR),
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Sales_SalesOrderHeaderSalesReason (
    SalesOrderID NUMBER NOT NULL,
    SalesReasonID NUMBER NOT NULL,
    ModifiedDate DATE NOT NULL,
    PRIMARY KEY (SalesOrderID, SalesReasonID)
);

CREATE TABLE Sales_SalesPerson (
    BusinessEntityID NUMBER PRIMARY KEY,
    TerritoryID NUMBER,
    SalesQuota NUMBER(12,2),
    Bonus NUMBER(12,2),
    CommissionPct NUMBER(5,4),
    SalesYTD NUMBER(14,2),
    SalesLastYear NUMBER(14,2),
    rowguid VARCHAR2(36 CHAR),
    ModifiedDate DATE
);

CREATE TABLE Sales_SalesPersonQuotaHistory (
    BusinessEntityID NUMBER NOT NULL,
    QuotaDate DATE NOT NULL,
    SalesQuota NUMBER(12,2),
    rowguid VARCHAR2(36 CHAR),
    ModifiedDate DATE NOT NULL,
    PRIMARY KEY (BusinessEntityID, QuotaDate)
);

CREATE TABLE Sales_SalesReason (
    SalesReasonID NUMBER PRIMARY KEY,
    Name VARCHAR2(50 CHAR) NOT NULL,
    ReasonType VARCHAR2(50 CHAR) NOT NULL,
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Sales_SalesTaxRate (
    SalesTaxRateID NUMBER PRIMARY KEY,
    StateProvinceID NUMBER NOT NULL,
    TaxType NUMBER(2) NOT NULL,
    TaxRate NUMBER(6,4) NOT NULL,
    Name VARCHAR2(50 CHAR) NOT NULL,
    rowguid VARCHAR2(36 CHAR),
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Sales_SalesTerritory (
    TerritoryID NUMBER PRIMARY KEY,
    Name VARCHAR2(50 CHAR) NOT NULL,
    CountryRegionCode VARCHAR2(3 CHAR) NOT NULL,
    "Group" VARCHAR2(50 CHAR) NOT NULL,
    SalesYTD NUMBER(14,2),
    SalesLastYear NUMBER(14,2),
    CostYTD NUMBER(14,2),
    CostLastYear NUMBER(14,2),
    rowguid VARCHAR2(36 CHAR),
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Sales_SalesTerritoryHistory (
    BusinessEntityID NUMBER NOT NULL,
    TerritoryID NUMBER NOT NULL,
    StartDate DATE NOT NULL,
    EndDate DATE,
    rowguid VARCHAR2(36 CHAR),
    ModifiedDate DATE NOT NULL,
    PRIMARY KEY (BusinessEntityID, TerritoryID, StartDate)
);

CREATE TABLE Sales_ShoppingCartItem (
    ShoppingCartItemID NUMBER PRIMARY KEY,
    ShoppingCartID NUMBER NOT NULL,
    Quantity NUMBER(5) NOT NULL,
    ProductID NUMBER NOT NULL,
    DateCreated DATE NOT NULL,
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Sales_SpecialOffer (
    SpecialOfferID NUMBER PRIMARY KEY,
    Description VARCHAR2(255 CHAR) NOT NULL,
    DiscountPct NUMBER(5,4) NOT NULL,
    Type VARCHAR2(50 CHAR) NOT NULL,
    Category VARCHAR2(50 CHAR) NOT NULL,
    StartDate DATE NOT NULL,
    EndDate DATE,
    MinQty NUMBER(5) NOT NULL,
    MaxQty NUMBER(5),
    rowguid VARCHAR2(36 CHAR),
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Sales_SpecialOfferProduct (
    SpecialOfferID NUMBER NOT NULL,
    ProductID NUMBER NOT NULL,
    rowguid VARCHAR2(36 CHAR),
    ModifiedDate DATE NOT NULL,
    PRIMARY KEY (SpecialOfferID, ProductID)
);

CREATE TABLE Purchasing_PurchaseOrderHeader (
    PurchaseOrderID NUMBER PRIMARY KEY,
    RevisionNumber NUMBER(3) NOT NULL,
    Status NUMBER(2) NOT NULL,
    EmployeeID NUMBER NOT NULL,
    VendorID NUMBER NOT NULL,
    ShipMethodID NUMBER NOT NULL,
    OrderDate DATE NOT NULL,
    ShipDate DATE,
    SubTotal NUMBER(14,2) NOT NULL,
    TaxAmt NUMBER(12,2) NOT NULL,
    Freight NUMBER(12,2) NOT NULL,
    TotalDue NUMBER(14,2) NOT NULL,
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Purchasing_PurchaseOrderDetail (
    PurchaseOrderID NUMBER NOT NULL,
    PurchaseOrderDetailID NUMBER PRIMARY KEY,
    DueDate DATE NOT NULL,
    OrderQty NUMBER(5) NOT NULL,
    ProductID NUMBER NOT NULL,
    UnitPrice NUMBER(12,2) NOT NULL,
    LineTotal NUMBER(14,2),
    ReceivedQty NUMBER(5),
    RejectedQty NUMBER(5),
    StockedQty NUMBER(5),
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Purchasing_ProductVendor (
    ProductID NUMBER NOT NULL,
    BusinessEntityID NUMBER NOT NULL,
    AverageLeadTime NUMBER(5),
    StandardPrice NUMBER(12,2) NOT NULL,
    LastReceiptCost NUMBER(12,2),
    LastReceiptDate DATE,
    MinOrderQty NUMBER(5),
    MaxOrderQty NUMBER(5),
    OnOrderQty NUMBER(5),
    UnitMeasureCode VARCHAR2(3 CHAR),
    ModifiedDate DATE NOT NULL,
    PRIMARY KEY (ProductID, BusinessEntityID)
);

CREATE TABLE Purchasing_ShipMethod (
    ShipMethodID NUMBER PRIMARY KEY,
    Name VARCHAR2(50 CHAR) NOT NULL,
    ShipBase NUMBER(12,2) NOT NULL,
    ShipRate NUMBER(12,2) NOT NULL,
    rowguid VARCHAR2(36 CHAR),
    ModifiedDate DATE NOT NULL
);

CREATE TABLE Purchasing_Vendor (
    BusinessEntityID NUMBER PRIMARY KEY,
    AccountNumber VARCHAR2(25 CHAR) NOT NULL,
    Name VARCHAR2(100 CHAR) NOT NULL,
    CreditRating NUMBER(2) NOT NULL,
    PreferredVendorStatus CHAR(1) NOT NULL,
    ActiveFlag CHAR(1) NOT NULL,
    PurchasingWebServiceURL VARCHAR2(255 CHAR),
    ModifiedDate DATE NOT NULL
);
CREATE TABLE UnitMeasure (
    UnitMeasureCode   VARCHAR2(3)    PRIMARY KEY,
    Name              VARCHAR2(50)   NOT NULL,
    ModifiedDate      DATE           NOT NULL
);

CREATE TABLE Production_WorkOrder (
    WorkOrderID       NUMBER PRIMARY KEY,
    ProductID         NUMBER NOT NULL,
    OrderQty          NUMBER NOT NULL,
    StockedQty        NUMBER NOT NULL,
    ScrappedQty       NUMBER DEFAULT 0,
    StartDate         DATE NOT NULL,
    EndDate           DATE,
    DueDate           DATE NOT NULL,
    ScrapReasonID     NUMBER,
    ModifiedDate      DATE DEFAULT SYSDATE
);

CREATE TABLE Production_TransactionHistoryArchive (
    TransactionID          NUMBER PRIMARY KEY,
    ProductID              NUMBER NOT NULL,
    ReferenceOrderID       NUMBER,
    ReferenceOrderLineID   NUMBER,
    TransactionDate        DATE NOT NULL,
    TransactionType        VARCHAR2(1) NOT NULL,
    Quantity               NUMBER NOT NULL,
    ActualCost             NUMBER(10,2),
    ModifiedDate           DATE DEFAULT SYSDATE
);

CREATE TABLE TransactionHistory (
    TransactionID          NUMBER PRIMARY KEY,
    ProductID              NUMBER NOT NULL,
    ReferenceOrderID       NUMBER,
    ReferenceOrderLineID   NUMBER,
    TransactionDate        DATE NOT NULL,
    TransactionType        VARCHAR2(1) NOT NULL CHECK (TransactionType IN ('P', 'S', 'W')),
    Quantity               NUMBER NOT NULL,
    ActualCost             NUMBER(10,2),
    ModifiedDate           DATE DEFAULT SYSDATE
);

CREATE TABLE Production_ScrapReason (
    ScrapReasonID   NUMBER PRIMARY KEY,
    Name            VARCHAR2(100) NOT NULL,
    ModifiedDate    DATE DEFAULT SYSDATE
);

CREATE TABLE Production_ProductSubcategory (
    ProductSubcategoryID   NUMBER PRIMARY KEY,
    ProductCategoryID      NUMBER NOT NULL,
    Name                   VARCHAR2(100) NOT NULL,
    rowguid                VARCHAR2(36) NOT NULL,
    ModifiedDate           DATE DEFAULT SYSDATE
);

