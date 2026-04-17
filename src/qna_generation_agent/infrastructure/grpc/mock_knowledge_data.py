"""Realistic mock Knowledge Service responses with proper educational content."""

from __future__ import annotations

from typing import Any

# Realistic educational content for OOP concepts - manually written for quality
OOP_CHUNKS = {
    "Encapsulation": [
        {
            "chunk_id": "chunk-encap-001",
            "content": "Encapsulation is the bundling of data (attributes) and methods (functions) that operate on that data within a single unit called a class. It is one of the four fundamental Object-Oriented Programming concepts, alongside inheritance, polymorphism, and abstraction. Encapsulation restricts direct access to some of an object's components, which prevents accidental interference and misuse of the data. This is typically achieved by declaring class fields as private and providing public getter and setter methods to access and modify them. For example, in a BankAccount class, the balance field would be private, and only deposit() and withdraw() methods could modify it, ensuring the balance cannot become negative through direct manipulation.",
            "source_type": "document",
            "metadata": {
                "page": "45",
                "source": "OOP_Fundamentals.pdf",
                "topic": "Encapsulation",
            },
        },
        {
            "chunk_id": "chunk-encap-002",
            "content": "Common implementation mistakes with encapsulation include: First, making fields public instead of private, which allows any external code to modify object state directly without validation. Second, creating getters that return references to mutable internal objects (like returning a Date object directly) instead of returning copies or immutable views, allowing external code to modify internal state indirectly. Third, writing setters without input validation, such as accepting negative values for age or allowing empty strings for required fields. Fourth, exposing internal collections directly instead of returning unmodifiable wrappers or defensive copies. These mistakes break encapsulation and lead to fragile, hard-to-maintain code that is difficult to debug.",
            "source_type": "document",
            "metadata": {
                "page": "47",
                "source": "OOP_Fundamentals.pdf",
                "topic": "Encapsulation",
                "focus": "common_errors",
            },
        },
        {
            "chunk_id": "chunk-encap-003",
            "content": 'Java example of proper encapsulation: public class Employee { private String name; private double salary; private Date hireDate; public Employee(String name, double salary) { setName(name); setSalary(salary); this.hireDate = new Date(); } public String getName() { return name; } public void setName(String name) { if (name == null || name.trim().isEmpty()) { throw new IllegalArgumentException("Name cannot be empty"); } this.name = name; } public double getSalary() { return salary; } public void setSalary(double salary) { if (salary < 0) { throw new IllegalArgumentException("Salary cannot be negative"); } this.salary = salary; } public Date getHireDate() { return new Date(hireDate.getTime()); } } This example demonstrates validation in constructors and setters, defensive copying for mutable objects (Date), and controlled access to internal state.',
            "source_type": "document",
            "metadata": {
                "page": "49",
                "source": "OOP_Fundamentals.pdf",
                "topic": "Encapsulation",
                "language": "Java",
            },
        },
    ],
    "Polymorphism": [
        {
            "chunk_id": "chunk-poly-001",
            "content": "Polymorphism is an Object-Oriented Programming concept that allows objects of different classes to be treated as objects of a common superclass. The term comes from Greek meaning 'many forms.' In programming, polymorphism allows methods to do different things based on the object it is acting upon, even though they share the same name. There are two main types: compile-time polymorphism (method overloading) where multiple methods share the same name but have different parameters, and runtime polymorphism (method overriding) where a subclass provides a specific implementation of a method already defined in its superclass. Runtime polymorphism is achieved through inheritance and interface implementation, enabling dynamic method dispatch where the method that gets executed is determined at runtime based on the actual object's class, not the reference type.",
            "source_type": "document",
            "metadata": {
                "page": "78",
                "source": "OOP_Fundamentals.pdf",
                "topic": "Polymorphism",
            },
        },
        {
            "chunk_id": "chunk-poly-002",
            "content": "Runtime polymorphism example in Java: Consider a Shape superclass with a draw() method, and subclasses Circle, Rectangle, and Triangle that each override draw() with their specific implementation. When you write Shape s = new Circle(); s.draw(); the JVM determines at runtime that s refers to a Circle object and calls Circle's draw() method, not Shape's. This enables writing generic code like: List<Shape> shapes = Arrays.asList(new Circle(), new Rectangle(), new Triangle()); for (Shape shape : shapes) { shape.draw(); } Each shape draws itself correctly without the calling code knowing the specific type. Common mistakes include: confusing method overloading with overriding (overloading is compile-time, overriding is runtime), forgetting the @Override annotation which can lead to subtle bugs if the method signature doesn't match, and attempting to override static methods (they hide instead, which is different).",
            "source_type": "document",
            "metadata": {
                "page": "82",
                "source": "OOP_Fundamentals.pdf",
                "topic": "Polymorphism",
                "focus": "runtime_examples",
            },
        },
        {
            "chunk_id": "chunk-poly-003",
            "content": "Benefits of polymorphism include code reusability and extensibility. With polymorphism, you can write generic methods that work with superclass references, and they automatically work with any future subclasses without modification. For example, a payment processing system can work with PaymentMethod superclass, and new payment types (CryptoPayment, BNPLPayment) can be added later without changing the processing code. Polymorphism also enables design patterns like Strategy (interchangeable algorithms), Factory (interchangeable object creation), and Command (interchangeable requests). Without polymorphism, you would need switch statements or if-else chains checking object types, leading to code that violates the Open/Closed Principle and is difficult to maintain. Polymorphism works hand-in-hand with inheritance to provide flexible, extensible designs.",
            "source_type": "document",
            "metadata": {
                "page": "85",
                "source": "OOP_Fundamentals.pdf",
                "topic": "Polymorphism",
                "focus": "benefits",
            },
        },
    ],
    "Abstraction": [
        {
            "chunk_id": "chunk-abs-001",
            "content": "Abstraction is the concept of hiding complex implementation details and showing only essential features to the user. It helps manage complexity by providing a simplified view of complex systems. Abstraction focuses on what an object does rather than how it does it. In Java, abstraction is achieved through abstract classes and interfaces. An abstract class cannot be instantiated and may contain abstract methods (methods without implementation) that must be implemented by subclasses. An interface is a completely abstract class that specifies a contract - a set of methods that implementing classes must provide. Real-world examples include: a car driver uses the steering wheel, accelerator, and brake without understanding the engine, transmission, or fuel injection systems; a smartphone user interacts with apps without knowing the underlying operating system, drivers, or hardware architecture.",
            "source_type": "document",
            "metadata": {
                "page": "102",
                "source": "OOP_Fundamentals.pdf",
                "topic": "Abstraction",
            },
        },
        {
            "chunk_id": "chunk-abs-002",
            "content": "Differences between abstract classes and interfaces: Abstract classes can have both abstract and concrete methods, instance variables, constructors, and can use any access modifier (public, protected, private). They represent an 'is-a' relationship and are used when classes share common implementation. Interfaces can only have abstract methods (until Java 8), static final constants (implicitly), and no constructors. All methods are implicitly public. Interfaces represent a 'can-do' relationship and are used when classes share common behavior but not implementation. Since Java 8, interfaces can have default and static methods with implementation. Since Java 9, interfaces can have private methods. A class can extend only one abstract class but can implement multiple interfaces, solving the multiple inheritance problem. Common mistakes: trying to instantiate abstract classes or interfaces directly, forgetting to implement all abstract methods in a subclass, and confusing when to use abstract class vs interface.",
            "source_type": "document",
            "metadata": {
                "page": "107",
                "source": "OOP_Fundamentals.pdf",
                "topic": "Abstraction",
                "focus": "abstract_vs_interface",
            },
        },
    ],
    "Inheritance": [
        {
            "chunk_id": "chunk-inherit-001",
            "content": "Inheritance is an Object-Oriented Programming mechanism where a new class (subclass, derived class, or child class) derives properties and behaviors from an existing class (superclass, base class, or parent class). It promotes code reusability and establishes a natural hierarchical classification. The subclass inherits fields and methods from the superclass and can add new fields and methods or override existing ones to specialize behavior. In Java, inheritance is declared using the 'extends' keyword for classes: public class Dog extends Animal { ... }. The subclass gets automatic access to all non-private members of the superclass. Constructors are not inherited but can be called using super(). The Object class is the ultimate superclass of all classes in Java. If a class does not explicitly extend another class, it implicitly extends Object.",
            "source_type": "document",
            "metadata": {
                "page": "32",
                "source": "OOP_Fundamentals.pdf",
                "topic": "Inheritance",
            },
        },
        {
            "chunk_id": "chunk-inherit-002",
            "content": "Types of inheritance: Single inheritance (one subclass extends one superclass) is the only type directly supported in Java. Multilevel inheritance (A extends B, C extends B) is allowed - a chain of inheritance. Hierarchical inheritance (multiple subclasses extend one superclass) is allowed. Multiple inheritance (one subclass extending multiple superclasses) is NOT allowed in Java to avoid the Diamond Problem (ambiguity when two superclasses have methods with the same signature). However, multiple inheritance of type is achieved through interfaces - a class can implement multiple interfaces. The super keyword is used to access superclass members: super.method() calls superclass method, super.field accesses superclass field (if accessible), super() calls superclass constructor (must be first statement in subclass constructor). Method overriding allows a subclass to provide specific implementation of a method already defined in its superclass. The overriding method must have the same name, return type, and parameters. Use @Override annotation to ensure you are actually overriding and not creating a new method.",
            "source_type": "document",
            "metadata": {
                "page": "36",
                "source": "OOP_Fundamentals.pdf",
                "topic": "Inheritance",
                "focus": "types_and_super",
            },
        },
        {
            "chunk_id": "chunk-inherit-003",
            "content": "Common inheritance pitfalls and best practices: Do not use inheritance just for code reuse - use composition instead when there is no 'is-a' relationship. For example, a Car 'has-a' Engine (composition), but a Dog 'is-an' Animal (inheritance). The Liskov Substitution Principle states that objects of a superclass shall be replaceable with objects of the subclass without affecting program correctness. Violations include: subclass throwing exceptions the superclass doesn't, subclass strengthening preconditions (requiring more), or subclass weakening postconditions (promising less). Avoid deep inheritance hierarchies (more than 3-4 levels) as they become hard to understand and maintain. Use the final keyword to prevent inheritance when appropriate: final class cannot be extended, final method cannot be overridden. Consider using the template method pattern when you want subclasses to customize specific steps of an algorithm defined in the superclass.",
            "source_type": "document",
            "metadata": {
                "page": "41",
                "source": "OOP_Fundamentals.pdf",
                "topic": "Inheritance",
                "focus": "best_practices",
            },
        },
    ],
}


def get_mock_topics(workflow_id: str) -> list[dict[str, Any]]:
    """Return mock topics for testing."""
    return [
        {
            "topic_id": f"topic-{workflow_id}-oop",
            "name": "Object-Oriented Programming",
            "subtopics": [
                {
                    "topic_id": f"sub-{workflow_id}-encap",
                    "name": "Encapsulation",
                    "subtopics": [],
                },
                {
                    "topic_id": f"sub-{workflow_id}-poly",
                    "name": "Polymorphism",
                    "subtopics": [],
                },
                {
                    "topic_id": f"sub-{workflow_id}-abs",
                    "name": "Abstraction",
                    "subtopics": [],
                },
                {
                    "topic_id": f"sub-{workflow_id}-inherit",
                    "name": "Inheritance",
                    "subtopics": [],
                },
            ],
        }
    ]


def get_mock_chunks(
    query: str, workflow_id: str, top_k: int = 3
) -> list[dict[str, Any]]:
    """Return realistic mock chunks for a query (subtopic name)."""
    # Find chunks matching the query/subtopic
    for subtopic, chunks in OOP_CHUNKS.items():
        if subtopic.lower() in query.lower() or query.lower() in subtopic.lower():
            return [
                {
                    "chunk_id": c["chunk_id"],
                    "workflow_id": workflow_id,
                    "content": c["content"],
                    "source_type": c["source_type"],
                    "metadata": c["metadata"],
                    "score": 0.95 - (i * 0.03),  # Descending scores
                }
                for i, c in enumerate(chunks[:top_k])
            ]

    # Default fallback with realistic generic content
    return [
        {
            "chunk_id": "chunk-default-001",
            "workflow_id": workflow_id,
            "content": f"{query} is an important concept in Object-Oriented Programming. Understanding this concept is essential for writing maintainable, reusable code. Students often struggle with the theoretical aspects initially, but practical examples and hands-on coding exercises help solidify understanding. Key aspects include proper syntax, common use cases, and integration with other OOP principles.",
            "source_type": "document",
            "metadata": {"topic": query, "fallback": "true"},
            "score": 0.75,
        }
    ]
